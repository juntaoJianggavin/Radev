"""RADAR 器官嵌入 (36×256) -> LLM 输入空间 (36×D_llm) 的 projector。

形状照 Hulu-Med 的 `build_mlp`：`Linear(256, D) -> GELU -> Linear(D, D)`（plan 步骤 5）。
与它不同的三处，都是 Jev 数据的形状要求的：

1. **固定 36 个槽位**。`ORGANS`（`radar_encoder.py`）的顺序就是特征矩阵的行序 ——
   无论这 36 行来自 stage 1 缓存 npz（`cache.py`）还是训练进程里现算
   （`visual_frontend.py`），行号 `i` 与器官一一对应。缺失的器官**不从序列里删掉** ——
   用可学的 `missing` 向量占位：槽位本身是位置信号，删掉它等于让「36 个槽」变成
   变长序列，后面的 org id 嵌入就对不上了（stage 1 里缺失器官是常态：
   一个腹部 CT 不会同时显示 36 个器官）。
2. **每槽加一个器官 id 嵌入**（`nn.Embedding(36, D)`）。缓存里的向量是 CLIP 式的
   器官嵌入（L2 归一化），但 36 个槽谁是谁不带语义 —— 器官身份只能由这个嵌入提供。
   `has_feat=False` 的槽上 id 嵌入照样加，所以「缺这个器官」与「有这个器官但特征弱」
   是两种不同的输入。
3. **一条 study 一次前向**：`[B,36,256] -> [B,36,D]`，36 个槽全给 LLM，不按题做注意力
   限制。plan §3.4 的 `radar_organ -> section -> disease` 映射表（哪些题该只看哪几个
   槽）是 v2 的事 —— 题面 instructions 里带器官英文名（`criteria.ORGAN_EN`），
   v1 让模型自己从题面里找器官。
4. **`has_whole=True` 时多一个第 37 槽**：RADAR+（`radar_ft` 身份）的整图分支向量
   —— 一个 token 池化全部 patch，无滑窗（`radar_encoder.infer_whole`）。它排在器官行
   之后，走同一个 MLP 与同一套槽位 id 嵌入，所以「整图」对模型来说就是第 37 个槽位；
   它与器官槽不同源（预处理窗宽都不一样，见 `volume.py`），因此 id 嵌入必须存在 ——
   不然模型分不清第 37 行是「又一个器官」还是「全图摘要」。

5. **`index=False` 时把槽位身份整个撤掉**（A1 消融「anatomically indexed organ ledger」）：
   第 2 条那个 `nn.Embedding(n_tok, D)` 换成**一个共享向量**加到每个槽上。这不是「删掉
   一个可学偏置」那么轻 —— 单个共享向量被 MLP 末层的 bias 完全吸收，所以对模型就是
   「没有任何一列携带器官身份」；身份的另一半（第 20 行永远是肝这种**固定行序**）由
   `dataset.py` 的逐 study 置换（`--slot-perm`）撤掉，两半都在才算这条臂。36 行向量、
   MLP、`missing` 占位一律不动 —— 撤的是索引，不是内容。

🔴 本模块**不读缓存、不读权重**：输入是 numpy 都能给的张量，`dataset.py` /
`visual_frontend.py` 负责把 npz 或现算结果读进来。dtype 由调用方决定（训练用 bf16
前向、参数 fp32）。
"""
import torch
import torch.nn as nn

from .radar_encoder import N_ORGANS

FEAT_DIM = 256


class OrganProjector(nn.Module):
    """`[B,n_tok,256]` + `[B,n_tok]` 掩膜 -> `[B,n_tok,D_llm]`；`n_tok = 36 (+1 整图)`。"""

    def __init__(self, d_llm: int, n_organs: int = N_ORGANS, d_in: int = FEAT_DIM,
                 has_whole: bool = False, index: bool = True):
        super().__init__()
        self.n_organs = int(n_organs)
        self.has_whole = bool(has_whole)
        self.index = bool(index)
        self.n_tok = self.n_organs + (1 if self.has_whole else 0)
        self.d_in = int(d_in)
        self.d_llm = int(d_llm)
        self.missing = nn.Parameter(torch.empty(d_in))
        if self.index:
            self.organ_emb = nn.Embedding(self.n_tok, self.d_llm)
            self.organ_shared = None
        else:
            self.organ_emb = None
            self.organ_shared = nn.Parameter(torch.empty(self.d_llm))
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_llm),
            nn.GELU(),
            nn.Linear(d_llm, d_llm),
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.missing, std=0.02)
        if self.index:
            nn.init.normal_(self.organ_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.organ_shared, std=0.02)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, feats: torch.Tensor, has_feat: torch.Tensor) -> torch.Tensor:
        """feats `[B,n_tok,256]`（float16/32均可），has_feat `[B,n_tok]` bool。"""
        if feats.shape[-2:] != (self.n_tok, self.d_in):
            raise ValueError(f'feats 形状 {tuple(feats.shape)} != (B,{self.n_tok},{self.d_in})')
        if has_feat.shape != feats.shape[:2]:
            raise ValueError(f'has_feat 形状 {tuple(has_feat.shape)} != '
                             f'{tuple(feats.shape[:2])}')
        w = self.mlp[0].weight
        x = feats.to(w.dtype)
        x = torch.where(has_feat.unsqueeze(-1), x, self.missing.to(w.dtype))
        h = self.mlp(x)
        add = (self.organ_emb.weight if self.index else self.organ_shared).to(h.dtype)
        return h + add.unsqueeze(0)

    def trainable_param_names(self):
        out = ['missing',
               'organ_emb.weight' if self.index else 'organ_shared']
        for i, m in enumerate(self.mlp):
            if isinstance(m, nn.Linear):
                out += [f'mlp.{i}.weight', f'mlp.{i}.bias']
        return out
