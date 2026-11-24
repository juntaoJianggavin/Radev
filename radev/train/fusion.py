"""B 支跨编码器融合：切片 token（Q）× RADAR 器官向量（K/V）的单点交叉注意力。

移植自 ClinFusion 的 CaSL/⊗3D 核心机制（`custom_model/modeling_multi_enc_qwen3_vl.py`
的 wrapper 级后处理：MHA + 残差加 + FFN）；砍掉它的 deepstack 多层重跑、局部 k×k
unfold 窗、随机残差正则与多专家级联 —— 我们只有一对编码器，37 个键也不需要窗口
（取舍清单见 `模型架构与训练计划.md` §9）。

三条实现约定，每条都有测试盯着（`analysis/smoke_q35_b.py` [7]）：

1. **零初始化 ⇒ 初始恒等**。`attn.out_proj` 与 `ffn` 末层的 weight/bias 全 0：
   初始时 `h = x + 0 + 0`（fp32 下逐位相等）。FUSE=1 与 FUSE=0 两条消融臂从同一
   函数出发，对照里不掺初始化噪声。
2. **逐样本调用**（model 侧按行拆）。一个 batch 里各题的器官向量不同，绝不能跨题
   互查 —— 这是形状上最容易错、错了还不报错的地方。B 很小且只有 37 个键，
   循环的代价可忽略，换掉 ragged 分组与逐 token 掩膜体操。
3. **fp32 算、bf16 送**（与 `OrganProjector` 同一条惯例）：参数恒 fp32，内部升到
   fp32 计算，输出 cast 回输入的 dtype。缺特征（`kv_mask` 全 False）恒等旁路；
   被掩的键在投影**之前**清零，防 `0 × NaN` 污染输出。
"""
import torch
import torch.nn as nn

from .projector import FEAT_DIM


def organ2d_agg(ie: torch.Tensor, tower_rows: list, grid: torch.Tensor,
                slice_z: torch.Tensor, organ_box: torch.Tensor, *,
                patch: int, merge: int, w0: int = 256, h0: int = 384,
                n_organs: int, has_whole: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """切片视觉塔输出 -> 每器官一格的 2D 聚合 `[n_tok, D]`（含整图槽）。

    归属规则（全部确定性、零学习参数 —— 与 `--fusion-pool` 的学习式注意力池化相对立，
    那条臂实测 AUC≈0.5；这里器官身份来自 `organ_box` 的空间归属，不来自 query 学习）：

    * merged cell `(r, c)` 的中心映回原始切片坐标：
      `y = (r+0.5)·merge·w0/h_patch`、`x = (c+0.5)·merge·h0/w_patch`
      （`h_patch/w_patch` 是 grid_thw 的后两位，乘 patch 即 processor resize 后的边长；
      原始切片恒 `w0×h0` = volume.load_and_prepare 的 padded 平面）；
    * 器官 o 的行 `(z0,z1,y0,y1,x0,x1)`（含界，-1 行 = 无该类）收下所有
      `z∈[z0,z1] 且 (y,x) 落盒` 的 cell，均值池化；
    * 整图槽（`has_whole` 时的第 37 行）= 全部 cell 的均值；
    * 无 cell 的器官（切片采样没盖到 z 范围 / 平面盒在图外 / box 缺列给 -1）→ 0 行 +
      `valid=False`（`Organ2DResidual` 里 proj 无 bias，0 行贡献恰为 0，不造假证据）。

    `ie [ΣL, D]` 按 `tower_rows` 逐样本切；`grid [Σn_img, 3]` 是本 batch 全部图的
    grid（每样本的行段），`slice_z [B, n_slices]` 是每片在 padded 体积里的层号。
    单样本内所有切片同尺寸 ⇒ grid 行必须全同（断言），t 必须为 1（切片是单帧图）。
    """
    dev = ie.device
    n_tok = n_organs + (1 if has_whole else 0)
    out = torch.zeros((len(tower_rows), n_tok, ie.shape[1]), device=dev,
                      dtype=torch.float32)
    valid = torch.zeros((len(tower_rows), n_tok), dtype=torch.bool, device=dev)
    g0 = lo = 0
    for b, L in enumerate(tower_rows):
        n_img = int(slice_z.shape[1])
        g = grid[g0:g0 + n_img]
        g0 += n_img
        if not bool((g == g[0]).all()):
            raise ValueError(f'样本 {b} 的 {n_img} 片 grid 不全同 —— organ2d 的坐标映射'
                             ' 假定同尺寸切片（off 口径的均匀采样保证这一点）')
        t, h_p, w_p = (int(v) for v in g[0])
        if t != 1:
            raise ValueError(f'样本 {b} 的切片 grid t={t} != 1 —— organ2d 只认单帧图')
        rows, cols = h_p // merge, w_p // merge
        if rows * cols * n_img != int(L):
            raise ValueError(f'样本 {b}：视觉塔 {int(L)} 行 != {n_img}×{rows}×{cols}'
                             ' —— grid 与 tower_rows 不同源')
        part = ie[lo:lo + int(L)].to(torch.float32)
        lo += int(L)
        n_cells = rows * cols
        idx = torch.arange(n_cells, device=dev)
        r, c = idx // cols, idx % cols
        y = ((r + 0.5) * merge * w0 / h_p).unsqueeze(0).expand(n_img, -1)
        x = ((c + 0.5) * merge * h0 / w_p).unsqueeze(0).expand(n_img, -1)
        z = slice_z[b].to(dev).unsqueeze(1).expand(-1, n_cells)
        y, x, z = y.reshape(-1), x.reshape(-1), z.reshape(-1)
        flat = part.reshape(n_img * n_cells, -1)
        box = organ_box[b].to(dev)
        for o in range(n_organs):
            bx = box[o]
            if int(bx.min()) < 0:
                continue
            m = ((z >= bx[0]) & (z <= bx[1]) & (y >= bx[2]) & (y <= bx[3])
                 & (x >= bx[4]) & (x <= bx[5]))
            if bool(m.any()):
                out[b, o] = flat[m].mean(dim=0)
                valid[b, o] = True
        if has_whole:
            out[b, n_organs] = flat.mean(dim=0)
            valid[b, n_organs] = flat.shape[0] > 0
    return out, valid


class Organ2DResidual(nn.Module):
    """`--organ2d`：projector 器官嵌入 + α·proj(逐器官 2D 聚合) 的门控残差。

    与池化臂（`SliceOrganPool`）的三点对立，每点都来自那条臂的实测失败（AUC 0.49）：
    ① 归属是**几何的**（organ_box 空间判定），不是学习式注意力 —— 器官身份不靠学；
    ② 聚合输出**不进序列**（序列保持 direct3d 的 37 槽不变），只作器官嵌入的残差
    修正 —— 2D 只能加分不能顶替；
    ③ **零初始化**（proj 无 bias 全 0 + 逐槽 α 初始化 0）⇒ 初始态与 organ-only 逐位
    相等，FUSE 消融从同一点出发；训练后 α 的分布直接回答「2D 有没有用」。
    proj 无 bias 是有意的：无 cell 的器官聚合行是 0，无 bias 才能保证它的贡献恰为 0。
    """

    def __init__(self, d_llm: int, n_tok: int, patch: int = 14, merge: int = 2):
        super().__init__()
        if patch < 1 or merge < 1:
            raise ValueError(f'patch={patch} / merge={merge} 都要 >=1')
        self.d_llm = int(d_llm)
        self.n_tok = int(n_tok)
        self.patch, self.merge = int(patch), int(merge)
        self.proj = nn.Linear(d_llm, d_llm, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.gate = nn.Parameter(torch.zeros(self.n_tok))

    def forward(self, organ_v: torch.Tensor, agg: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        """`organ_v [B,n_tok,D] fp32`（projector 输出）；`agg/valid` 见 `organ2d_agg`
        （同一 B）。返回同形状 —— `organ_v + α ⊙ proj(agg)`。"""
        if organ_v.shape[:2] != agg.shape[:2] or organ_v.shape[2] != agg.shape[2]:
            raise ValueError(f'organ_v {tuple(organ_v.shape)} 与 agg {tuple(agg.shape)} '
                             '对不上（B/n_tok/D 必须一致）')
        g = self.gate.to(torch.float32).unsqueeze(0).unsqueeze(-1)
        return organ_v + g * self.proj(agg.to(torch.float32))

    def trainable_param_names(self):
        return ['proj.weight', 'gate']


class OrganRoute(nn.Module):
    """`--organ-route`：题绑定器官槽的**软门控**（零初始化恒等起点）。

    与 organ_map（题名 -> 绑定器官）配合，把稀疏的 RADAR 视觉信号聚焦到与当前题
    因果相关的那个器官：对绑定器官槽**上调**、其余**下调**（不硬删 —— 多器官题会
    丢上下文），逐样本做均值居中的乘性门控（k 只走前 n_organs 个器官槽）：

        scale[b,k] = 1 + tanh(gamma) · (m[b,k] − mean_k m[b,k])

    三点家法（与 `Organ2DResidual` 同源）：
    ① **零初始化**：gamma=0 ⇒ tanh=0 ⇒ scale≡1 ⇒ 初始态与 organ-only 逐位相等，
       路由消融从同一点出发；训练后 gamma 的取值直接回答「路由有没有用」。
    ② **均值居中**：平均 scale≈1（范数近似守恒），绑定槽 >1、非绑定槽 <1；无绑定
       （m 全 0）⇒ centered 全 0 ⇒ 恒等，自动退回 organ-only。少器官题（绑 1/36）
       的不对称是有意的 —— 非绑定槽只被轻微下调，保住上下文。
    ③ **整图中性**：只缩放前 n_organs 个器官槽，第 37 整图槽（若有）scale 恒 1 ——
       它是全局摘要，不是「又一个 competing 器官」，不该被逐题门控下调。
    与 direct3d/fuse/organ2d/evidence/pool 全正交：各臂改的是槽的**值**怎么算，
    路由在其后（`model._embed` 的 scatter 之前）统一缩放。只有 1 个标量参数 gamma。
    """

    kind = 'scalar'

    def __init__(self, n_organs: int):
        super().__init__()
        self.n_organs = int(n_organs)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, v: torch.Tensor, route_mask: torch.Tensor) -> torch.Tensor:
        """`v [B,n_tok,D]`（projector/各臂汇合后的器官槽值）；`route_mask [B,n_tok]`
        bool（绑定器官槽 True，只有前 n_organs 槽可能 True）。返回同形状。"""
        if route_mask is None:
            return v
        m = route_mask.to(torch.float32)[:, :self.n_organs]
        centered = m - m.mean(dim=1, keepdim=True)
        g = torch.tanh(self.gamma.to(torch.float32))
        scale = 1.0 + g * centered
        if v.shape[1] > self.n_organs:
            neutral = scale.new_ones(v.shape[0], v.shape[1] - self.n_organs)
            scale = torch.cat([scale, neutral], dim=1)
        return v * scale.unsqueeze(-1).to(v.dtype)

    def trainable_param_names(self):
        return ['gamma']

    def summary(self):
        return {'mode': self.kind, 'gamma': float(self.gamma.detach().float())}


class OrganRouteVisual(nn.Module):
    """`--organ-route-visual`：每器官一个**视觉条件**软门控（零初始化恒等起点）。

    与标量版共用同一套路由语义（绑定槽上调 / 其余下调、均值居中、整图中性），
    只把「一个全局 gamma」换成「每个器官自己的权重」：

        a[b,k]    = bias[k] + (proj[k] · v[b,k]) / sqrt(D)
        scale[b,k] = 1 + tanh(a[b,k]) · (m[b,k] − mean_j m[b,j])

    再对 `tanh(a)·centered` 做一次样本内去均值，保证 36 个 scale 的均值恒为 1
    （与标量版同一条「平均音量守恒」）。

    每个器官的权重有两个来源：
    * `bias[k]`：这个器官的静态先验（肝是否普遍比脾更该信）；
    * `proj[k] · v_k`：这个器官**当前这张图**的视觉特征 —— 同一器官在不同图上可以
      拿到不同权重（例如某图肝槽特征很弱/很噪，就把它的门控压下去）。
    `bias` 全相等且 `proj=0` 时严格退化为标量版 OrganRoute，所以它是标量版的严格
    超集，不是另一条无关的臂；step 0 两者与不开路由都逐位相等。

    与 direct3d/fuse/organ2d/evidence/pool 全正交（作用在 `_embed` 里各臂汇合后的
    器官槽值上），不增删 token，整图第 37 槽仍中性。
    """

    kind = 'visual'

    def __init__(self, n_organs: int, d_llm: int):
        super().__init__()
        self.n_organs = int(n_organs)
        self.d_llm = int(d_llm)
        self.bias = nn.Parameter(torch.zeros(self.n_organs))
        self.proj = nn.Parameter(torch.zeros(self.n_organs, self.d_llm))
        self._inv_sqrt_d = float(self.d_llm) ** -0.5

    def forward(self, v: torch.Tensor, route_mask: torch.Tensor) -> torch.Tensor:
        """`v [B,n_tok,D]`；`route_mask [B,n_tok]` bool（前 n_organs 槽可能 True）。"""
        if route_mask is None:
            return v
        m = route_mask.to(torch.float32)[:, :self.n_organs]
        centered = m - m.mean(dim=1, keepdim=True)
        vo = v[:, :self.n_organs].to(torch.float32)
        a = self.bias.to(torch.float32).unsqueeze(0) + torch.einsum(
            'bkd,kd->bk', vo, self.proj.to(torch.float32)) * self._inv_sqrt_d
        g = torch.tanh(a)
        delta = g * centered
        delta = delta - delta.mean(dim=1, keepdim=True)
        scale = 1.0 + delta
        if v.shape[1] > self.n_organs:
            neutral = scale.new_ones(v.shape[0], v.shape[1] - self.n_organs)
            scale = torch.cat([scale, neutral], dim=1)
        return v * scale.unsqueeze(-1).to(v.dtype)

    def trainable_param_names(self):
        return ['bias', 'proj']

    def summary(self):
        p = self.proj.detach().float()
        return {'mode': self.kind,
                'bias': self.bias.detach().float().tolist(),
                'proj_norm': float(p.norm()),
                'proj_row_norm': p.norm(dim=1).tolist()}


class OrganRoutePerOrgan(nn.Module):
    """`--organ-route-perorgan`：**每器官一个静态标量**的题绑定软门控。

    与标量版共用同一套路由语义（绑定槽上调 / 其余下调、均值居中、整图中性、
    无绑定恒等），只把「一个全局 gamma」换成「36 个各管各的 gamma」：

        g[k]       = tanh(bias[k])                  # 每个器官自己的权重
        delta[b,k] = g[k] · (m[b,k] − mean_j m[b,j])
        delta[b,k] -= mean_j delta[b,j]             # 平均 scale 恒为 1
        scale[b,k] = 1 + delta[b,k]

    与 visual 版的区别只在「权重从哪来」：这里 g[k] 是**静态**的（不看这张图的视觉
    特征），visual 版是 `bias[k] + proj[k]·v[b,k]`。所以本版是标量版的严格超集
    （36 个 bias 全相等 ⇔ 标量版），参数量 36，没有 proj。无绑定 / bias 全零时
    与不开路由逐位相等；第 37 整图槽仍中性。
    """

    kind = 'perorgan'

    def __init__(self, n_organs: int):
        super().__init__()
        self.n_organs = int(n_organs)
        self.bias = nn.Parameter(torch.zeros(self.n_organs))

    def forward(self, v: torch.Tensor, route_mask: torch.Tensor) -> torch.Tensor:
        """`v [B,n_tok,D]`；`route_mask [B,n_tok]` bool（前 n_organs 槽可能 True）。"""
        if route_mask is None:
            return v
        m = route_mask.to(torch.float32)[:, :self.n_organs]
        centered = m - m.mean(dim=1, keepdim=True)
        g = torch.tanh(self.bias.to(torch.float32)).unsqueeze(0)
        delta = g * centered
        delta = delta - delta.mean(dim=1, keepdim=True)
        scale = 1.0 + delta
        if v.shape[1] > self.n_organs:
            neutral = scale.new_ones(v.shape[0], v.shape[1] - self.n_organs)
            scale = torch.cat([scale, neutral], dim=1)
        return v * scale.unsqueeze(-1).to(v.dtype)

    def trainable_param_names(self):
        return ['bias']

    def summary(self):
        b = self.bias.detach().float()
        g = torch.tanh(b)
        return {'mode': self.kind,
                'bias': b.tolist(),
                'gamma_mean': float(g.mean()),
                'gamma_std': float(g.std(unbiased=False)),
                'gamma_min': float(g.min()),
                'gamma_max': float(g.max())}


class OrganAcquire(nn.Module):
    """`--organ-acquire`（armEV）：每器官一个**证据获取策略**打分头（零初始化中性起点）。

    与 `OrganRouteVisual` 同参数化（`bias[k] + proj[k]·v_k`），但语义相反：路由是**门控**
    （缩放器官槽的值，恒等起点），获取是**策略**（输出 Bernoulli logit，决定这个器官的
    证据要不要取）。armEV 的 RL 动作 = 「取哪些器官的证据」：本头给出每个器官的获取
    logit，loop 侧 Bernoulli 采样子集 -> 确定性掩码 -> 答案前向 -> GRPO 回传本头。

        a[b,k]  = bias[k] + (proj[k]·v[b,k])/sqrt(D) + route_w·route_mask[b,k]  (k<n_organs)
        a[b,36] = whole_bias                                                    (整图槽)

    四点家法：
    ① **零初始化**：bias/proj/route_w/whole_bias 全 0 ⇒ logit≡0 ⇒ σ=0.5 中性探索，
       不预设「该看哪」；策略从均匀探索出发，由奖励塑造（与路由的恒等起点同源纪律）。
    ② **视觉条件**：`proj[k]·v_k` 让同一器官在不同图上拿不同获取分（这张图肝槽特征弱
       就少取）；`bias[k]` 是器官静态先验。
    ③ **题绑定偏好**：`route_w·route_mask` 给题面绑定器官加一个可学习偏置（初始 0）。
    ④ **整图槽单独一个标量** `whole_bias`：整图是全局摘要不是「又一个器官」，不做逐图
       视觉条件；它值不值得取由奖励决定（`acq_whole_frac` 盯着整图捷径坍缩）。
    缺特征器官（`has_feat=False`）的视觉项清零（只留 bias），且 loop 侧 `present` 掩码
    把它们排除出采样 —— 不在场的器官不可获取。输出是**策略 logit 不是门控**，fp32 算。
    """

    kind = 'acquire'

    def __init__(self, n_organs: int, d_llm: int):
        super().__init__()
        self.n_organs = int(n_organs)
        self.d_llm = int(d_llm)
        self.bias = nn.Parameter(torch.zeros(self.n_organs))
        self.proj = nn.Parameter(torch.zeros(self.n_organs, self.d_llm))
        self.route_w = nn.Parameter(torch.zeros(()))
        self.whole_bias = nn.Parameter(torch.zeros(()))
        self._inv_sqrt_d = float(self.d_llm) ** -0.5

    def forward(self, v: torch.Tensor, route_mask: torch.Tensor,
                has_feat: torch.Tensor) -> torch.Tensor:
        """`v [B,n_tok,D]`（projector 器官槽，n_tok=n_organs+1 含整图）；
        `route_mask [B,n_tok]` bool（绑定器官 True，可 None）；`has_feat [B,n_tok]` bool。
        返回 `[B,n_organs+1]` 获取 logit（前 n_organs 器官 + 末位整图）。"""
        B = v.shape[0]
        vo = v[:, :self.n_organs].to(torch.float32)
        hf = has_feat[:, :self.n_organs].to(torch.float32)
        visual = torch.einsum('bkd,kd->bk', vo,
                              self.proj.to(torch.float32)) * self._inv_sqrt_d
        visual = visual * hf
        a = self.bias.to(torch.float32).unsqueeze(0) + visual
        if route_mask is not None:
            m = route_mask.to(torch.float32)[:, :self.n_organs]
            a = a + self.route_w.to(torch.float32) * m
        whole = self.whole_bias.to(torch.float32).expand(B, 1)
        return torch.cat([a, whole], dim=1)

    def trainable_param_names(self):
        return ['bias', 'proj', 'route_w', 'whole_bias']

    def summary(self):
        p = self.proj.detach().float()
        return {'mode': self.kind,
                'bias': self.bias.detach().float().tolist(),
                'proj_norm': float(p.norm()),
                'route_w': float(self.route_w.detach().float()),
                'whole_bias': float(self.whole_bias.detach().float())}


def organ_region_mask(grid_rows: torch.Tensor, slice_z_b: torch.Tensor,
                      box_row: torch.Tensor, *, merge: int,
                      w0: int = 256, h0: int = 384) -> torch.Tensor:
    """一个器官的 box -> 该样本视觉塔输出行上的 bool 掩码 `[L]`。

    与 `organ2d_agg` 同一条坐标数学（cell 中心映回 padded 平面 + z 用层号），但不做
    均值池化 —— 返回**命中的行集**，供证据 token 的 K/V 取用（`--organ-evidence`）。
    `grid_rows [n_img,3]` 是该样本各片的 grid（须全同、t=1）；`slice_z_b [n_img]` 是
    各片层号；`box_row [6]`（含界，-1 行由调用方先筛掉）。全 -1 / 无命中 → 全 False。
    """
    if int(box_row.min()) < 0:
        return None
    if not bool((grid_rows == grid_rows[0]).all()):
        raise ValueError('organ_region_mask 假定同尺寸切片（off/z 口径保证）')
    t, h_p, w_p = (int(v) for v in grid_rows[0])
    if t != 1:
        raise ValueError(f'切片 grid t={t} != 1 —— 证据臂只认单帧图')
    rows, cols = h_p // merge, w_p // merge
    n_img = int(slice_z_b.shape[0])
    n_cells = rows * cols
    dev = slice_z_b.device
    idx = torch.arange(n_cells, device=dev)
    r, c = idx // cols, idx % cols
    y = (r + 0.5) * merge * w0 / h_p
    x = (c + 0.5) * merge * h0 / w_p
    z0, z1, y0, y1, x0, x1 = (int(v) for v in box_row[:6])
    m = ((slice_z_b.unsqueeze(1) >= z0) & (slice_z_b.unsqueeze(1) <= z1)
         & (y.unsqueeze(0) >= y0) & (y.unsqueeze(0) <= y1)
         & (x.unsqueeze(0) >= x0) & (x.unsqueeze(0) <= x1)).reshape(-1)
    return m


class OrganEvidence(nn.Module):
    """`--organ-evidence`：题绑定器官的 box 区域切片 token -> 一个证据 token。

    Q=该器官的 projector 嵌入，K/V=落在它 box 里的切片 token（`organ_region_mask`
    的几何归属 —— 身份来自几何不来自学习，避开池化臂 AUC≈0.5 的死因）；输出
    `q + out(Attn(q, region))`，`out` 零初始化 ⇒ **初始 token = 器官向量本身**
    （信息量为零的起点：拼接臂没有残差臂的逐位恒等，这是它最强的等价纪律）。
    区域为空（切片没盖到 z 范围 / 平面盒在图外 / box 缺列）→ 直接返回 q，不造假证据。
    fp32 算、调用方 cast 回 LLM dtype（与 projector 同一条惯例）。
    """

    def __init__(self, d_llm: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if d_llm % heads:
            raise ValueError(f'd_llm={d_llm} 不被 heads={heads} 整除')
        self.d_llm = int(d_llm)
        self.attn = nn.MultiheadAttention(d_llm, heads, dropout=dropout,
                                          batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q `[D]`（该器官的 projector 嵌入）；kv `[m, D]`（区域 token，m 可为 0）。
        返回 `[D]`。按 attention 权重的 dtype 算 —— 训练在 autocast 下 q/kv 是 bf16，
        eval 不走 autocast、kv 仍是 bf16 而权重是 fp32；不在这里对齐就会
        `mat1 and mat2 must have the same dtype`（2026-11-02 2B oev eval 实测）。"""
        if kv.shape[0] == 0:
            return q
        w = self.attn.in_proj_weight
        q = q.to(w.dtype)
        kv = kv.to(w.dtype)
        out, _ = self.attn(q.view(1, 1, -1), kv.unsqueeze(0), kv.unsqueeze(0),
                           need_weights=False)
        return q + out.squeeze(0).squeeze(0)

    def trainable_param_names(self):
        names = ['attn.in_proj_weight', 'attn.in_proj_bias',
                 'attn.out_proj.weight', 'attn.out_proj.bias']
        return names


class SliceOrganFusion(nn.Module):
    """`[L,D_llm]` 切片 token + `[n_tok,256]` 器官向量 -> `[L,D_llm]`。

    `n_tok = 36 (+1 整图)` 与 `OrganProjector.n_tok` 同口径，但 K/V 走**独立的**
    k/v 投影，不借道 projector（它的输出是为「进 LLM 词嵌入空间」校准的，融合是
    另一条用法；两件事共用一个参数化会把梯度语义搅在一起）。
    """

    def __init__(self, d_llm: int, d_kv: int = FEAT_DIM, heads: int = 8,
                 dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        if d_llm % heads:
            raise ValueError(f'd_llm={d_llm} 不被 heads={heads} 整除'
                             f'（d_llm: 0.8B=1024 / 2B=2048 / 4B=2560，tiny=64，都应当整除）')
        self.d_llm = int(d_llm)
        self.heads = int(heads)
        self.k_proj = nn.Linear(d_kv, d_llm)
        self.v_proj = nn.Linear(d_kv, d_llm)
        self.attn = nn.MultiheadAttention(d_llm, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_llm)
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, ffn_mult * d_llm),
            nn.GELU(),
            nn.Linear(ffn_mult * d_llm, d_llm),
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        nn.init.zeros_(self.ffn[2].weight)
        nn.init.zeros_(self.ffn[2].bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor) -> torch.Tensor:
        """x `[L, D_llm]`（一片样本的切片 token）；kv `[n_tok, 256]`；
        kv_mask `[n_tok]` bool（`has_feat` 那一行）。"""
        if not bool(kv_mask.any()):
            return x
        kv = torch.where(kv_mask.unsqueeze(-1), kv.to(torch.float32),
                         torch.zeros((), device=kv.device, dtype=torch.float32))
        x32 = x.to(torch.float32)
        out, _ = self.attn(x32.unsqueeze(0),
                           self.k_proj(kv).unsqueeze(0),
                           self.v_proj(kv).unsqueeze(0),
                           key_padding_mask=(~kv_mask).unsqueeze(0),
                           need_weights=False)
        h = x32 + out.squeeze(0)
        h = h + self.ffn(self.ln(h))
        return h.to(x.dtype)

    def trainable_param_names(self):
        names = ['k_proj.weight', 'k_proj.bias', 'v_proj.weight', 'v_proj.bias',
                 'attn.in_proj_weight', 'attn.in_proj_bias', 'attn.out_proj.weight',
                 'attn.out_proj.bias', 'ln.weight', 'ln.bias']
        for i, m in enumerate(self.ffn):
            if isinstance(m, nn.Linear):
                names += [f'ffn.{i}.weight', f'ffn.{i}.bias']
        return names


class SliceOrganPool(nn.Module):
    """器官条件池化：`[L,D]` 切片 token -> `[n_tok,D]` 摘要 token。

    存在理由是**稀释**，不是又一层融合：8 片 × 143 个视觉 token = 1,144 个 token 对 36/37
    个器官槽（约 31:1，3D 只占 3.2%），语言先验可以轻易把视觉证据压过去。这里用 n_tok 个
    器官 query 对全部切片 token 做一次交叉注意力，只把 n_tok 个摘要送进 LLM —— 视觉序列从
    1,144 缩到 36 个摘要（另加原样的 36/37 个器官槽）。

    `n_tok` 是**器官数**（36），不是序列里的器官槽数（radar_ft 有整图槽，是 37）：
    整图槽不是器官，不产生 summary query；它照常作为器官槽进序列。这样摘要与器官
    一一对应，序列结构是「36 摘要 + 36/37 器官槽」。

    与 `SliceOrganFusion` 的分工：那边是**切片读器官**（把器官先验注进每个切片 token，Q=切片），
    这边是**器官读切片**（每个器官从全部切片里取证据，Q=器官）。两者可以串联（本臂就是
    FUSE=1 + POOL=N），也可以只用池化；池化本身不要求融合先跑。

    Query 直接取 `OrganProjector` 输出的**前 n_tok 行**（器官槽，不含整图槽）：
    * 摘要 token 与器官槽同源同序，第 k 个摘要就是「第 k 个器官看到的切片证据」；
    * 缺器官时 projector 已把它换成 `missing` 向量，query 仍然存在（「没有这个器官」
      本身是证据），不会因为 has_feat=False 就少一个摘要；
    * 不另学一套 query 投影会少一组随机初始化，但也少了「摘要与器官槽解耦」的自由度 ——
      这是有意的取舍：我们要的是把器官证据送进 LLM，不是再造一个自由记忆槽。

    与融合臂的零初始化约定相反，这里**不**零初始化 `out_proj/ffn`：池化的输出不是残差
    恒等（没有「输入即输出」可保），零初始化会让 q/k/v 在第一步拿不到梯度（梯度要穿过
    out_proj 的 0 权重），等于池化永远学不动。初始态是「器官嵌入 + 切片平均」——注意力
    均匀时正是这个语义，梯度从一开始就能分流。
    """

    def __init__(self, d_llm: int, n_tok: int, heads: int = 8,
                 dropout: float = 0.0, ffn_mult: int = 4):
        super().__init__()
        if d_llm % heads:
            raise ValueError(f'd_llm={d_llm} 不被 heads={heads} 整除')
        if n_tok < 1:
            raise ValueError(f'n_tok={n_tok} < 1')
        self.d_llm = int(d_llm)
        self.n_tok = int(n_tok)
        self.heads = int(heads)
        self.q_proj = nn.Linear(d_llm, d_llm)
        self.k_proj = nn.Linear(d_llm, d_llm)
        self.v_proj = nn.Linear(d_llm, d_llm)
        self.attn = nn.MultiheadAttention(d_llm, heads, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_llm)
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, ffn_mult * d_llm),
            nn.GELU(),
            nn.Linear(ffn_mult * d_llm, d_llm),
        )

    def forward(self, x: torch.Tensor, q_base: torch.Tensor) -> torch.Tensor:
        """x `[L,D]`（一片样本的切片 token，可为融合后）；q_base `[n_tok,D]`（器官槽嵌入）。

        返回 `[n_tok,D]`，dtype 跟 x。`L=0`（没有切片）时直接返回 q_base —— 与融合的
        「缺卷旁路」同一条约定：没有证据可取时，摘要就是器官本身，不造一个假的平均。
        """
        if x.shape[0] == 0:
            return q_base.to(x.dtype)
        x32 = x.to(torch.float32)
        q = self.q_proj(q_base.to(torch.float32)).unsqueeze(0)
        out, _ = self.attn(q, self.k_proj(x32).unsqueeze(0),
                           self.v_proj(x32).unsqueeze(0), need_weights=False)
        h = q + out
        h = h + self.ffn(self.ln(h))
        return h.squeeze(0).to(x.dtype)

    def trainable_param_names(self):
        names = ['q_proj.weight', 'q_proj.bias', 'k_proj.weight', 'k_proj.bias',
                 'v_proj.weight', 'v_proj.bias', 'attn.in_proj_weight', 'attn.in_proj_bias',
                 'attn.out_proj.weight', 'attn.out_proj.bias', 'ln.weight', 'ln.bias']
        for i, m in enumerate(self.ffn):
            if isinstance(m, nn.Linear):
                names += [f'ffn.{i}.weight', f'ffn.{i}.bias']
        return names
