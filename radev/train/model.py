"""组装：RADAR 器官嵌入 -> projector -> 小 LLM -> 受限 softmax 读出。

计划出处（plan 步骤 5 / §3.2、§3.3）。受限 softmax 的候选集与答案文本由
`pack.py` 一处定义（a13 单射、a14 候选 == criteria），本模块只负责打分。

**两条打分路径，训练一条、评估一条**，差别只在候选前缀：

* `forward()` —— 训练/监控。序列里带 ground-truth 答案（teacher forcing），**一次前向**
  同时读出全部候选的分数。候选 k 的第 j 个 token 由位置 `answer_start + j - 1` 的
  logprob 读取（causal shift：位置 p 的 logits 预测 p+1，所以答案首 token 由
  `answer_start - 1` 处的 logits 预测）。
  * j = 0：所有候选共享同一个位置 ⇒ 首 token 分数**精确**；
  * j >= 1：上下文是 **target 自己的前缀** ⇒ 目标候选精确，其余候选拿到的是
    「target 前缀下的条件概率」。这是单前向的固有近似，取舍写在这里：共享前缀的候选
    被精确比较（多 token choice 族要的正是这个 —— 它们首 token 常常相同），
    首 token 就不同的候选在 j = 0 处已经分开。
* `score_exact()` —— 评估。每个候选配**自己的前缀**（prompt 复制 K 份），分数是
  `sum_j log P(tok_j | prompt + tok_<j)` 的精确值，无 teacher forcing、无泄漏。
  **报数只走这条**；`forward()` 的分数只当训练监控（与推理条件差着上面那条近似）。

两条路径的候选 id 都来自 `dataset` 的 `EncodedSpace.cand_ids`（`pack.encode_space`），
包括「前导空格」这条编码约定 —— 这里不碰 tokenizer。

**B 支：两条视觉通道并存**（宿主换成原生多模态的 Qwen3.5 之后）：

* **原生通道**：CT 切片当普通图片喂给宿主自带的 `visual` 塔。嵌入由宿主自己的
  `get_image_features(pixel_values, image_grid_thw)` 出，写进 `input_ids` 里
  `image_token_id`（实测 248056）占的那些位置 —— 这一步在宿主 `forward` 里也有
  （modeling_qwen3_5.py:1652-1661），但我们不能走它：那条路要求 `input_ids` 与
  `inputs_embeds` **二选一**（:1646 的异或检查），而 RADAR 通道必须自己填 `inputs_embeds`。
* **RADAR 通道**：器官槽仍由我们的 `OrganProjector` 出，占位 token 用词表里的**空闲 id**
  （实测 248077 起共 243 个，解码为空串）⇒ 不需要 `resize_token_embeddings`，
  宿主词表与存盘权重一字不动。
* **位置编码是这条路唯一的硬约束**：M-RoPE 只在 `input_ids` + `mm_token_type_ids` +
  多模态数据三者齐全时才算（:1583-1612）。我们喂的是 `inputs_embeds`（`input_ids=None`）
  ⇒ 宿主会静默退化成 `attention_mask.cumsum` 的 1D 位置，图像段与文本段的 3D 位置全错。
  所以 `_positions()` 用宿主自己的 `get_rope_index` 算出 `[3,B,T]` 再显式传 `position_ids`
  —— 传了 `position_ids`，`compute_3d_position_ids` 整段被跳过（:1674）。

**池化臂（`--fusion-pool N`）**是 B 支的第三种视觉接线：序列里不再有「每片 t·h·w/merge²」
个图像占位符，而是 N 个摘要占位符（N = 器官数 36）。视觉塔仍按真实切片出 ΣL 行，`_embed`
里先（可选）过 `SliceOrganFusion`，再用 projector 的 36 个器官嵌入做 query、以全部切片
token 为 K/V 做交叉注意力池化，最后把 N 行摘要 scatter 进这 N 个位置。摘要对 LLM 就是
普通视觉嵌入，`_positions()` 因此返回 None（走宿主自己的 1D 位置）—— 切片网格与摘要数
本来就对不上，强行套 M-RoPE 只会错位。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import (SliceOrganFusion, SliceOrganPool, Organ2DResidual,
                     OrganEvidence, OrganRoute, OrganRouteVisual, OrganRoutePerOrgan,
                     OrganAcquire, organ2d_agg, organ_region_mask)
from .projector import FEAT_DIM, OrganProjector
from .radar_encoder import N_ORGANS

_DTYPES = {'bfloat16': torch.bfloat16, 'bf16': torch.bfloat16,
           'float32': torch.float32, 'fp32': torch.float32,
           'float16': torch.float16, 'fp16': torch.float16}


def _text_cfg(cfg):
    """取**文本侧** config：复合宿主（Qwen3.5 多模态 = text_config + vision_config）的
    顶层没有 `hidden_size`，直接读会 `AttributeError: 'Qwen3_5Config' object has no
    attribute 'hidden_size'`（实测 2026-10-20，0.8B：顶层无该键，text_config 是 1024）；
    扁平文本宿主（A 支的 Qwen3）自己就是文本 config，没有 `text_config` 属性。
    两条通道都从这里取，不按调用方声明分支。
    """
    tc = getattr(cfg, 'text_config', None)
    return cfg if tc is None else tc


def _vision_geom(cfg):
    """宿主视觉 config -> (patch, merge)。organ2d 的坐标映射要用（cell 中心怎么落回
    原始切片坐标），不能抄常数 —— 换宿主这两位数会变，错了是静默的归属错位。"""
    vc = getattr(cfg, 'vision_config', None)
    if vc is None:
        raise ValueError('宿主没有 vision_config —— organ2d 需要原生视觉塔的几何参数')
    return int(vc.patch_size), int(vc.spatial_merge_size)


def _cand_index(cand_ids, answer_start, device, answer_len=None):
    """ragged 候选 -> 四张稠密索引张量（`pos`/`tok`/`valid`/`cvalid`）+ 被剔除的候选下标。

    `pos[b,k,j] = answer_start[b] + j - 1`（causal shift 的读点），
    `valid[b,k,j] = j < len(候选 k)`，`cvalid[b,k] = 候选 k 存在`。
    无效位置填 0 并由 `valid`/`cvalid` 掩掉 —— 索引张量里不能有越界值。

    🔴 `answer_len`（这条样本真正被喂进上下文的答案 token 数）给定时：**比它长的候选
    整个剔除**（`cvalid=False`）。原因是硬的，不是防御性编程 —— teacher-forced 路径里
    只有目标答案的那几个位置被喂过，`j ≥ answer_len` 的读点落在 pad 位置上，取到的是
    「没有上下文的 logits」；而 pad 之后不越界时（batch 里别的样本更长）它连 IndexError
    都不报，静默把垃圾项加进分数。剔除而不是截断求和，是因为截断后的部分和与目标的
    完整和不可比（少几项负对数必然更大），argmax 会系统性偏向长候选。
    精确分数走 `score_exact()`（每个候选配自己的前缀），那才是报数用的路径。
    """
    B = len(cand_ids)
    if B == 0:
        raise ValueError('空 batch')
    k_max = max(len(c) for c in cand_ids)
    t_max = max(len(t) for c in cand_ids for t in c)
    pos = torch.zeros((B, k_max, t_max), dtype=torch.long)
    tok = torch.zeros((B, k_max, t_max), dtype=torch.long)
    valid = torch.zeros((B, k_max, t_max), dtype=torch.bool)
    cvalid = torch.zeros((B, k_max), dtype=torch.bool)
    dropped = []
    for b, cands in enumerate(cand_ids):
        p0 = int(answer_start[b])
        if p0 < 1:
            raise ValueError(f'样本 {b} 的 answer_start={p0} < 1：没有可用来预测首 token 的位置')
        lim = None if answer_len is None else int(answer_len[b])
        for k, ids in enumerate(cands):
            if not ids:
                raise ValueError(f'样本 {b} 的第 {k} 个候选是空 token 序列（pack.encode_space 应拦下）')
            if lim is not None and len(ids) > lim:
                dropped.append((b, k, len(ids), lim))
                continue
            cvalid[b, k] = True
            for j, t in enumerate(ids):
                pos[b, k, j] = p0 + j - 1
                tok[b, k, j] = int(t)
                valid[b, k, j] = True
    return (pos.to(device), tok.to(device), valid.to(device), cvalid.to(device),
            dropped)


class JevModel(nn.Module):
    """`llm`（冻结与否由调用方决定）+ `projector`（`[B,36,256] -> [B,36,D_llm]`）。

    序列里的视觉占位符（`vis_mask`，36 个/行）在 `_embed` 里被 projector 输出**整体覆盖**
    —— 占位 token 的取值只影响可读性，不影响前向。`vis_mask` 全空（state-only 地板，
    `n_vis=0`）时 projector 不参与，同一份代码跑 ① 级地板。

    宿主是原生多模态（有 `vision_config`）时多一条通道：`input_ids` 里的
    `image_token_id` 位置由宿主自己的 `visual` 塔填（B 支），见模块 docstring。
    """

    def __init__(self, llm, projector, *, pad_id=None, fusion=None, pool=None,
                 organ2d=None, evidence=None, route=None, ev_nobox=False,
                 acquire=None):
        super().__init__()
        self.llm = llm
        self.projector = projector
        self.fusion = fusion
        self.pool = pool
        self.organ2d = organ2d
        self.evidence = evidence
        self.route = route
        self.acquire = acquire
        if self.acquire is not None:
            if evidence is None:
                raise ValueError('organ-acquire 需要 organ-evidence —— 获取动作掩的是证据槽')
            if self.acquire.n_organs != projector.n_organs:
                raise ValueError(f'acquire 的器官数 {self.acquire.n_organs} != projector 的 '
                                 f'{projector.n_organs}')
        self.ev_merge = _vision_geom(llm.config)[1] if evidence is not None else None
        self.ev_nobox = bool(ev_nobox)
        if self.ev_nobox and evidence is None:
            raise ValueError('ev_nobox 要 evidence 模块 —— 没有证据 token 就没有 K/V 可放宽')
        cfg_pad = getattr(_text_cfg(llm.config), 'pad_token_id', None)
        if cfg_pad is None and pad_id is None:
            raise ValueError('文本侧 config 的 pad_token_id 是 None 且没给 pad_id —— '
                             '评估路径要它来填 padding（Qwen3.5 系必须由调用方传 pad_id）')
        self.pad_id = int(pad_id if pad_id is not None else cfg_pad)
        self.d_llm = projector.d_llm
        self.n_organs = projector.n_organs
        self.n_tok = projector.n_tok
        cfg = llm.config
        self.has_vision = getattr(cfg, 'vision_config', None) is not None
        self.image_token_id = int(getattr(cfg, 'image_token_id', -1))
        if self.has_vision and self.image_token_id < 0:
            raise ValueError('宿主有 vision_config 却没有 image_token_id —— '
                             '原生通道的插槽无处可放')
        if self.organ2d is not None:
            if not self.has_vision:
                raise ValueError('organ2d 需要宿主自带视觉塔（vision_config）—— '
                                 '纯文本宿主没有 2D 通路')
            if self.organ2d.n_tok != self.n_tok:
                raise ValueError(f'organ2d 的槽位数 {self.organ2d.n_tok} != 模型的 '
                                 f'{self.n_tok}（整图槽口径不一致）')
        if self.evidence is not None:
            if not self.has_vision:
                raise ValueError('organ-evidence 需要宿主自带视觉塔（vision_config）—— '
                                 '证据 token 的 K/V 来自 2D 塔输出')
            if self.organ2d is not None or self.pool is not None:
                raise ValueError('organ-evidence 与 organ2d/fusion-pool 互斥 —— '
                                 '三条臂都改器官段的序列/嵌入结构，只能开一条')

    @classmethod
    def from_pretrained(cls, model_path, *, n_organs=N_ORGANS, dtype='bfloat16',
                        freeze_llm=False, attn_implementation=None, pad_id=None,
                        has_whole=False, fuse=False, pool=0, organ2d=False,
                        evidence=False, route=False, route_visual=False,
                        route_perorgan=False, ev_nobox=False, index=True,
                        acquire=False):
        """加载 Qwen3.5 宿主 + 新建 projector（A/B 两支同一宿主，A 支不喂切片）。

        LLM 权重按 `dtype` 加载；**projector 恒为 fp32**（`_embed` 里把它的输出 cast 到
        LLM 的 dtype）。`freeze_llm=True` 时只有 projector 有梯度（全参训练是默认 ——
        plan 只冻结 RADAR，不冻结小 LLM）。`has_whole=True` 给 radar_ft 的整图分支多开
        一个槽位，必须与 `VisualFrontend.want_whole` / `dataset.n_rows` 一致。
        `fuse=True` 建 `SliceOrganFusion`（B 支 `--fuse`；恒 fp32、零初始化恒等起点 ——
        它挂在模型上，`freeze_llm` 只冻 `llm.parameters()`，所以对齐段它照常有梯度）。
        `index=False` 把 projector 的槽位身份嵌入换成单个共享向量（A1 消融 `--organ-index off`；
        固定行序那一半在 `dataset.py` 的 `--slot-perm` 那边撤）。

        **宿主类按 config 选**，不按调用方声明：有 `vision_config` 就走
        `AutoModelForImageTextToText`（Qwen3.5 系，B 支两条通道），没有就走
        `AutoModelForCausalLM`（纯文本基座，只有 RADAR 一条通道）。
        实测（2026-10-14，jjt_mllm / tf 5.7）：`AutoModelForCausalLM.from_config` 拿
        Qwen3.5 的 VL config 会当场 `AttributeError: 'Qwen3_5Config' has no attribute
        'vocab_size'` —— 它映射到 `Qwen3_5ForCausalLM`（纯文本类，读扁平 config）。
        所以「按 model_type 选类」不是风格问题，选错就载不进（`from_pretrained` 那条
        兼容路径能load 进 VL 权重，但那是映射表的事，不该依赖）。
        """
        from transformers import AutoConfig, AutoModelForCausalLM
        if isinstance(dtype, str):
            if dtype not in _DTYPES:
                raise SystemExit(f'dtype={dtype!r} 不在 {sorted(set(_DTYPES))}')
            dtype = _DTYPES[dtype]
        kw = {'attn_implementation': attn_implementation} if attn_implementation else {}
        cfg = AutoConfig.from_pretrained(model_path, **{k: v for k, v in kw.items()
                                                        if k == 'trust_remote_code'})
        if getattr(cfg, 'vision_config', None) is not None:
            from transformers import AutoModelForImageTextToText
            llm = AutoModelForImageTextToText.from_pretrained(model_path, dtype=dtype, **kw)
        else:
            llm = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kw)
        if freeze_llm:
            for p in llm.parameters():
                p.requires_grad_(False)
        h = int(_text_cfg(llm.config).hidden_size)
        projector = OrganProjector(h, n_organs=n_organs, has_whole=has_whole, index=index)
        fusion = SliceOrganFusion(h) if fuse else None
        pool_mod = (SliceOrganPool(h, projector.n_organs) if int(pool or 0) > 0 else None)
        if pool_mod is not None and int(pool) != projector.n_organs:
            raise ValueError(f'--fusion-pool={pool} 必须等于器官数 {projector.n_organs}'
                             '（每个器官一个 query，整图槽不产生摘要）')
        o2d = (Organ2DResidual(h, projector.n_tok, *_vision_geom(llm.config))
               if organ2d else None)
        ev_mod = OrganEvidence(h) if evidence else None
        if route_visual and route_perorgan:
            raise ValueError('route_visual 与 route_perorgan 互斥 —— 同一个门控只能有一种权重参数化')
        route_mod = None
        if route or route_visual or route_perorgan:
            if route_visual:
                route_mod = OrganRouteVisual(projector.n_organs, h)
            elif route_perorgan:
                route_mod = OrganRoutePerOrgan(projector.n_organs)
            else:
                route_mod = OrganRoute(projector.n_organs)
        if acquire and not evidence:
            raise ValueError('--organ-acquire 需要 --organ-evidence（获取动作掩的是证据槽）')
        acquire_mod = OrganAcquire(projector.n_organs, h) if acquire else None
        return cls(llm, projector, pad_id=pad_id, fusion=fusion, pool=pool_mod,
                   organ2d=o2d, evidence=ev_mod, route=route_mod, ev_nobox=ev_nobox,
                   acquire=acquire_mod)

    @classmethod
    def tiny(cls, tok, *, n_organs=N_ORGANS, hidden_size=64, num_hidden_layers=2,
             num_attention_heads=2, head_dim=32, intermediate_size=128, seed=0,
             dtype=torch.float32, has_whole=False, vl=False, vision_hidden=32,
             vocab_slack=256, fuse=False, pool=0, organ2d=False, evidence=False,
             route=False, route_visual=False, route_perorgan=False, ev_nobox=False,
             index=True, acquire=False):
        """CPU 冒烟专用：**真 tokenizer + 随机小权重**（Qwen3.5 `from_config`）。

        存在的理由是步骤 5 的验证不该依赖真实 LLM 权重 —— 它验的是前向数学、索引与
        受限 softmax，不是模型质量。真训练一律走 `from_pretrained`。

        `vl=True` 用随机初始化的 **Qwen3.5 多模态**宿主（B 支冒烟：两条通道 + M-RoPE
        的接线）；`vl=False` 用同代的 **Qwen3.5 纯文本**宿主（A 支冒烟，只有 RADAR
        一条通道）。两处不能想当然：

        * `vocab_size` 要 **> RADAR 槽的占位 id**（真模型里是词表尾部的空闲 id 248077）。
          只给 `len(tok)` 会让 `_embed` 的越界检查在 B 支当场炸 —— 所以留 `vocab_slack`。
        * `text_config.out_hidden_size` 必须 == 文本 `hidden_size`，否则原生通道的
          `masked_scatter` 维度对不上（真权重里 27B 是 5120、0.8B 实测 1024，各自自洽）。
        """
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        torch.manual_seed(seed)
        if vl:
            from transformers import Qwen3_5Config
            cfg = Qwen3_5Config(
                text_config=dict(vocab_size=len(tok) + vocab_slack, hidden_size=hidden_size,
                                 head_dim=head_dim, num_hidden_layers=num_hidden_layers,
                                 num_attention_heads=num_attention_heads,
                                 num_key_value_heads=num_attention_heads,
                                 intermediate_size=intermediate_size,
                                 linear_num_key_heads=2, linear_num_value_heads=2,
                                 linear_key_head_dim=32, linear_value_head_dim=32),
                vision_config=dict(depth=2, hidden_size=vision_hidden,
                                   intermediate_size=vision_hidden * 2, num_heads=4,
                                   out_hidden_size=hidden_size, num_position_embeddings=2304),
                tie_word_embeddings=False)
            cfg.pad_token_id = pad
            from transformers import AutoModelForImageTextToText
            llm = AutoModelForImageTextToText.from_config(cfg).to(dtype)
        else:
            from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
            cfg = Qwen3_5TextConfig(
                vocab_size=len(tok) + vocab_slack, hidden_size=hidden_size,
                head_dim=head_dim, num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_attention_heads,
                intermediate_size=intermediate_size, tie_word_embeddings=True,
                linear_num_key_heads=2, linear_num_value_heads=2,
                linear_key_head_dim=32, linear_value_head_dim=32)
            cfg.pad_token_id = pad
            llm = Qwen3_5ForCausalLM(cfg).to(dtype)
        proj = OrganProjector(hidden_size, n_organs=n_organs, has_whole=has_whole,
                              index=index)
        pool_mod = (SliceOrganPool(hidden_size, proj.n_organs) if int(pool or 0) > 0 else None)
        if pool_mod is not None and int(pool) != proj.n_organs:
            raise ValueError(f'pool={pool} 必须等于器官数 {proj.n_organs}')
        o2d = (Organ2DResidual(hidden_size, proj.n_tok, *_vision_geom(llm.config))
               if organ2d else None)
        if organ2d and not vl:
            raise ValueError('organ2d 只在 vl=True 的 tiny 冒烟里有意义（2D 塔来自宿主）')
        if evidence and not vl:
            raise ValueError('organ-evidence 只在 vl=True 的 tiny 冒烟里有意义（2D 塔来自宿主）')
        if route_visual and route_perorgan:
            raise ValueError('route_visual 与 route_perorgan 互斥 —— 同一个门控只能有一种权重参数化')
        route_mod = None
        if route or route_visual or route_perorgan:
            if route_visual:
                route_mod = OrganRouteVisual(proj.n_organs, hidden_size)
            elif route_perorgan:
                route_mod = OrganRoutePerOrgan(proj.n_organs)
            else:
                route_mod = OrganRoute(proj.n_organs)
        if acquire and not evidence:
            raise ValueError('acquire 需要 evidence（tiny 冒烟里两者都要 vl=True）')
        return cls(llm, proj, pad_id=pad, fusion=SliceOrganFusion(hidden_size) if fuse else None,
                   pool=pool_mod, organ2d=o2d,
                   evidence=OrganEvidence(hidden_size) if evidence else None,
                   route=route_mod, ev_nobox=ev_nobox,
                   acquire=OrganAcquire(proj.n_organs, hidden_size) if acquire else None)

    def radar_slot_id(self, n_tokenizer_tokens):
        """RADAR 槽该用的占位 id：词表里**没有对应字符串**的那一段。

        实测（Qwen3.5-0.8B，`analysis/probe_q35_proc.py`）：`len(tok)=248077` 而上限
        `vocab_size=248320` —— 尾部 243 个 id 解码为空串，任何 tokenizer 往返都产不出它们
        （探针里把 248077 塞进文本再 encode，回来是 0 个）。用它当器官槽的占位符，
        就不需要 `resize_token_embeddings`（那会动 embedding 形状，与存盘权重不再同源），
        三种插槽也各自唯一：文本 / 图像（248056）/ 器官（空闲 id）。
        纯文本宿主没有多模态插槽，`pad_id` 就够 —— 但这里的取值口径保持一条。
        """
        vocab = self.llm.get_input_embeddings().weight.shape[0]
        if int(n_tokenizer_tokens) >= vocab:
            return self.pad_id
        cand = int(n_tokenizer_tokens)
        if cand == self.image_token_id:
            raise ValueError(f'空闲段的第一个 id {cand} 正好是宿主的 image_token_id —— '
                             '两条通道的插槽会撞，换一个空闲 id')
        return cand

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_state_dict(self):
        """检查点只存有梯度的张量（冻结的 LLM 3.4 GB 不该进每个 epoch 的 ckpt）。"""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, sd):
        cur = self.state_dict()
        missing = [k for k, v in sd.items() if k not in cur or cur[k].shape != v.shape]
        if missing:
            raise ValueError(f'检查点里有 {len(missing)} 个键与当前模型对不上：{missing[:3]}')
        self.load_state_dict(sd, strict=False)

    def acquire_scores(self, batch):
        """armEV 获取策略打分：projector + `OrganAcquire`，**不过 LLM**。

        返回 `[B, n_organs+1]` 获取 logit（前 n_organs 器官 + 末位整图）。loop 侧用它
        Bernoulli 采样子集 -> 确定性掩码 -> 答案前向。只读 feats/has_feat/route_mask，
        不碰序列/像素 —— 获取决策在答案前向**之前**做（动作改的是输入，不是答案）。
        """
        if self.acquire is None:
            raise ValueError('acquire_scores 需要 --organ-acquire 头')
        feats, has = batch.get('feats'), batch.get('has_feat')
        if feats is None or has is None:
            raise ValueError('acquire_scores 缺 feats/has_feat')
        v32 = self.projector(feats, has)
        return self.acquire(v32, batch.get('route_mask'), has)

    def describe(self):
        n_tot = sum(p.numel() for p in self.parameters())
        n_tr = sum(p.numel() for p in self.trainable_parameters())
        cfg = self.llm.config
        route_summary = self.route.summary() if self.route is not None else None
        return {'llm_class': type(self.llm).__name__,
                'llm_layers': getattr(cfg, 'num_hidden_layers', None),
                'llm_frozen': not any(p.requires_grad for p in self.llm.parameters()),
                'd_llm': self.d_llm, 'n_organs': self.n_organs,
                'n_tok': self.n_tok, 'has_whole': self.projector.has_whole,
                'organ_index': self.projector.index,
                'projector_trainable': any(p.requires_grad for p in self.projector.parameters()),
                'has_vision': self.has_vision,
                'fuse': self.fusion is not None,
                'fusion_params': (sum(p.numel() for p in self.fusion.parameters())
                                  if self.fusion is not None else None),
                'fusion_pool': (self.pool.n_tok if self.pool is not None else 0),
                'pool_params': (sum(p.numel() for p in self.pool.parameters())
                                if self.pool is not None else None),
                'organ2d': self.organ2d is not None,
                'organ2d_params': (sum(p.numel() for p in self.organ2d.parameters())
                                   if self.organ2d is not None else None),
                'organ2d_gate': (self.organ2d.gate.detach().float().tolist()
                                 if self.organ2d is not None else None),
                'organ_evidence': self.evidence is not None,
                'evidence_no_box': bool(self.ev_nobox),
                'evidence_params': (sum(p.numel() for p in self.evidence.parameters())
                                    if self.evidence is not None else None),
                'organ_route': self.route is not None,
                'route_mode': (route_summary['mode'] if route_summary else None),
                'route_gamma': (route_summary.get('gamma') if route_summary else None),
                'route_bias': (route_summary.get('bias') if route_summary else None),
                'route_proj_norm': (route_summary.get('proj_norm')
                                    if route_summary else None),
                'route_summary': route_summary,
                'organ_acquire': self.acquire is not None,
                'acquire_params': (sum(p.numel() for p in self.acquire.parameters())
                                   if self.acquire is not None else None),
                'acquire_summary': (self.acquire.summary() if self.acquire is not None else None),
                'image_token_id': self.image_token_id if self.has_vision else None,
                'params_total': n_tot, 'params_trainable': n_tr,
                'projector_keys': self.projector.trainable_param_names()}

    def _native_embeds(self, pixel_values, image_grid_thw):
        """宿主自己的视觉塔：切片 -> 每图一段嵌入，按 grid 的拼接序摊平成 [ΣP', D]。

        与 modeling_qwen3_5.py:1652-1657 同一条链（`get_image_features` 的 `pooler_output`
        是被 `split_sizes` 切成的**元组**，必须 cat 回去才能喂 `masked_scatter`），
        但在这里调 —— 宿主 `forward` 里的那次调用要求 `input_ids`，而我们要自己填
        `inputs_embeds`（两者互斥，:1646）。
        """
        base = getattr(self.llm, 'model', self.llm)
        out = base.get_image_features(pixel_values, image_grid_thw, return_dict=True)
        ie = out.pooler_output
        return ie if torch.is_tensor(ie) else torch.cat(list(ie), dim=0)

    def _shared_native_embeds(self, pixel_values, image_grid_thw, pix_rows, n_img,
                              tower_rows, G):
        """armEV 视觉塔跨 G 份共享：每题只跑一次塔，特征按 G 重复展开到 B·G 行。

        题主序（行 `b·G+g` = 题 b 的第 g 个子集），G 份副本的 pixel_values/grid 逐位
        相同（`apply_acq_mask` 只改 has_feat/organ_box）；视觉塔 attention_dropout=0
        ⇒ 前向确定 ⇒ 共享与逐份前向**逐位同值**，repeat 的反向把 G 份梯度求和回同一
        份塔输出 ⇒ 梯度也与逐份前向累积一致。省 G× 塔激活显存（4B 全参+ref 的 OOM 主因）。
        只在 `ev_vision_group>1` 时由 `_embed` 调；既有臂与 eval（G=1）不走这里。
        """
        N = int(pix_rows.shape[0])
        if N % G != 0:
            raise ValueError(f'视觉共享：batch {N} 行须为 G={G} 的整数倍（题主序）')
        B = N // G
        pcum = torch.cumsum(pix_rows, 0)
        gcum = torch.cumsum(n_img, 0)
        pv_parts, grid_parts = [], []
        for b in range(B):
            r = b * G
            p0 = int(pcum[r - 1]) if r > 0 else 0
            g0 = int(gcum[r - 1]) if r > 0 else 0
            pv_parts.append(pixel_values[p0:int(pcum[r])])
            grid_parts.append(image_grid_thw[g0:int(gcum[r])])
        ie_q = self._native_embeds(torch.cat(pv_parts, 0), torch.cat(grid_parts, 0))
        lens_q = [int(tower_rows[b * G]) for b in range(B)]
        return torch.cat([s.repeat(G, 1) for s in ie_q.split(lens_q)], 0)

    def _fuse_native(self, ie, feats, has_feat, imask):
        """原生塔输出按样本拆开、逐样本过融合 —— 各题的器官向量不同，不能跨题互查。

        拆串用 `imask` 的逐行计数：它与 `masked_scatter` 的填充序同构，行数已由
        `_embed` 里的那条断言保证（`imask.sum() * D == ie.numel()`）。
        """
        parts = ie.split(imask.sum(dim=-1).tolist())
        outs = [self.fusion(p, feats[b], has_feat[b]) for b, p in enumerate(parts)]
        return torch.cat(outs, 0)

    def _evidence_embeds(self, ie, lens, grid, slice_z, organ_box, v32, ev_organs,
                         has_feat):
        """证据 token：逐样本、逐绑定器官（`fusion.OrganEvidence`）。

        返回 `[Σm, D_llm]` fp32，行序 = (样本序, ev_organs 升序) —— 与 pack 的交错占位
        序同构（`masked_scatter` 按行内下标递增填）。`v32 [B, n_tok, D]` 是 projector 输出
        （query 取其器官行）。区域为空的器官退回器官向量本身（零信息，不造假证据）。
        无绑定器官的题回退整图全局槽（`k == n_organs`，第 37 槽）：query=整图嵌入、
        K/V=该样本全部切片 token（无 box），与器官区域证据共用同一个 OrganEvidence。

        `has_feat`（armEV `--organ-acquire`）：获取掩码把未取槽的 has_feat 翻 False。
        整图槽没有 organ_box，has_feat 是它唯一的证据杠杆 —— 被屏的整图若仍走 cross-att
        会把全卷切片漏进来。仅 `self.acquire` 开着时启用这条回退；既有臂（acquire=None）
        逐位不变（它们的整图证据只在无绑定题上、且从不被 mask，见 views.plan_views）。
        """
        outs = []
        lo = g0 = 0
        n_img = int(slice_z.shape[1])
        for b, L in enumerate(lens):
            part = ie[lo:lo + int(L)]
            lo += int(L)
            g = grid[g0:g0 + n_img]
            g0 += n_img
            for k in ev_organs[b]:
                if self.acquire is not None and not bool(has_feat[b, k]):
                    outs.append(v32[b, k])
                    continue
                if k == self.n_organs:
                    outs.append(self.evidence(v32[b, k], part))
                    continue
                if self.ev_nobox:
                    outs.append(self.evidence(v32[b, k], part))
                    continue
                m = organ_region_mask(g, slice_z[b], organ_box[b][k],
                                      merge=self.ev_merge)
                if m is None or int(m.sum()) == 0:
                    outs.append(v32[b, k])
                    continue
                outs.append(self.evidence(v32[b, k], part[m]))
        return torch.stack([o.to(torch.float32) for o in outs], 0) if outs \
            else ie.new_zeros((0, v32.shape[-1]), dtype=torch.float32)

    def _embed(self, input_ids, vis_mask, feats, has_feat,
               pixel_values=None, image_grid_thw=None, tower_rows=None,
               organ_box=None, slice_z=None, ev_mask=None, ev_organs=None,
               route_mask=None, ev_vision_group=None, pix_rows=None, n_img=None):
        """词嵌入 ->（B 支）原生切片覆盖 `image_token_id` 位 -> RADAR 槽覆盖占位符位。

        `--fuse` 时切片 token 在 scatter 之前先过 `SliceOrganFusion`（Q=切片 token，
        K/V=器官向量）—— 训练与 `score_exact()` 都经这里，一处接线两条路自动一致。

        `--fusion-pool` 时走向另一条分支：切片段在序列里只有 `pool.n_tok` 个占位符，
        视觉塔仍按真实切片出 `ΣL` 行，先（可选）融合，再按样本用**器官 query（前 n_organs
        行，不含整图槽）**池化成 `pool.n_tok` 行，最后 scatter。`tower_rows` 是**每个样本的
        视觉塔输出行数**（由 `image_grid_thw` 现算，dataset 逐样本给）—— 不能用 `imask` 的
        行数拆，那个数已经是池化后的 N。

        `--organ2d`（第三条）：切片 token **不进序列**（imask 必须为空），视觉塔输出按
        `organ_box` 空间归属聚到器官粒度（`fusion.organ2d_agg`），经零初始化门控残差注入
        projector 输出 —— `v32` 在这里算好，下面器官槽的 scatter 路径原样使用。

        `--organ-evidence`（第四条）：切片 token 同样不进序列，但器官槽后各跟一个**证据
        占位符**（`ev_mask`，pack 按 ev_organs 交错拼的）—— 绑定器官的 box 区域切片 token
        经 cross-att 出一个 token 填进该位（`_evidence_embeds`）。器官槽本身照常走
        projector（这条臂不改 v32，证据是**加法**不是替换）。无绑定器官的题回退整图全局槽
        （第 37 槽）：query=整图嵌入、K/V=全部切片 token（见 `_evidence_embeds`）。

        两条通道的插槽**必须不相交**：原生通道按 `input_ids == image_token_id` 找位置，
        RADAR 通道按 `vis_mask`（空闲 id 248077 那一族）。相交意味着 pack.py 拼错了
        插槽，后果是后一次 scatter 静默盖掉前一次 —— 所以这里断言，不留口子。
        """
        emb = self.llm.get_input_embeddings()
        vocab = emb.weight.shape[0]
        mx = int(input_ids.max()) if input_ids.numel() else 0
        if mx >= vocab:
            raise ValueError(f'input_ids 里出现 {mx} >= 词表 {vocab} —— tokenizer 与权重不是同一套')
        x = emb(input_ids)
        v32 = None
        evs = None
        if pixel_values is not None:
            if not self.has_vision:
                raise ValueError('宿主是纯文本基座（没有 vision_config），但批次里带了 pixel_values '
                                 '—— --llm 与数据通路不一致')
            if image_grid_thw is None:
                raise ValueError('给了 pixel_values 没给 image_grid_thw：视觉塔不知道有几张图')
            if ev_vision_group is not None and int(ev_vision_group) > 1 \
                    and pix_rows is not None and n_img is not None:
                ie = self._shared_native_embeds(pixel_values, image_grid_thw,
                                                pix_rows, n_img, tower_rows,
                                                int(ev_vision_group))
            else:
                ie = self._native_embeds(pixel_values, image_grid_thw)
            ie = ie.to(x.device, x.dtype)
            imask = input_ids == self.image_token_id
            if self.evidence is not None:
                if int(imask.sum()) != 0:
                    raise ValueError(f'organ-evidence 臂的序列里有 {int(imask.sum())} 个图像'
                                     '占位符 —— 该臂切片 token 不进序列，pack 拼了逐片插槽'
                                     '就是接线错')
                if ev_mask is None or ev_organs is None:
                    raise ValueError('organ-evidence 开着但批次缺 ev_mask/ev_organs —— '
                                     '证据占位的位置与归属少一个就没法填')
                if organ_box is None or slice_z is None or tower_rows is None:
                    raise ValueError('organ-evidence 开着但批次缺 organ_box/slice_z/'
                                     'tower_rows —— 区域归属的三要素少一个')
                if feats is None or has_feat is None:
                    raise ValueError('organ-evidence 的 query 是 projector 器官嵌入 —— '
                                     'feats/has_feat 缺了')
                lens = [int(t) for t in tower_rows]
                if sum(lens) != ie.shape[0]:
                    raise ValueError(f'视觉塔输出 {ie.shape[0]} 行 != tower_rows 之和 '
                                     f'{sum(lens)}（image_grid_thw 与序列不是同一批切片）')
                n_ev = int(ev_mask.sum())
                per_row = [len(e) for e in ev_organs]
                if sum(per_row) != n_ev:
                    raise ValueError(f'ev_organs 共 {sum(per_row)} 个 != ev_mask 的 '
                                     f'{n_ev} 个占位符 —— pack 与 batch 的器官归属不同源')
                v32 = self.projector(feats, has_feat)
                evs = self._evidence_embeds(ie, lens, image_grid_thw, slice_z,
                                            organ_box, v32, ev_organs, has_feat)
                if bool((ev_mask & vis_mask).any()):
                    raise ValueError('证据占位与器官槽占位相交 —— 两种 id 用串了')
            elif self.organ2d is not None:
                if int(imask.sum()) != 0:
                    raise ValueError(f'organ2d 臂的序列里有 {int(imask.sum())} 个图像占位符 —— '
                                     '该臂切片 token 不进序列，pack 拼了逐片插槽就是接线错')
                if organ_box is None or slice_z is None or tower_rows is None:
                    raise ValueError('organ2d 开着但批次缺 organ_box/slice_z/tower_rows —— '
                                     '空间归属的三要素少一个就没法聚合')
                if feats is None or has_feat is None:
                    raise ValueError('organ2d 的残差基座是 projector 输出 —— feats/has_feat 缺了')
                lens = [int(t) for t in tower_rows]
                if sum(lens) != ie.shape[0]:
                    raise ValueError(f'视觉塔输出 {ie.shape[0]} 行 != tower_rows 之和 {sum(lens)}'
                                     '（image_grid_thw 与序列不是同一批切片）')
                agg, ovalid = organ2d_agg(
                    ie, lens, image_grid_thw, slice_z, organ_box,
                    patch=self.organ2d.patch, merge=self.organ2d.merge,
                    n_organs=self.n_organs, has_whole=self.projector.has_whole)
                if bool((~ovalid[:, :self.n_organs]).all()):
                    self.organ2d.no_box_hits = getattr(self.organ2d, 'no_box_hits', 0) + 1
                v32 = self.organ2d(self.projector(feats, has_feat), agg, ovalid)
            elif self.pool is not None:
                B, N = input_ids.shape[0], self.pool.n_tok
                if int(imask.sum()) != B * N:
                    raise ValueError(f'池化臂的序列里有 {int(imask.sum())} 个图像占位符，'
                                     f'期望 {B}×{N} —— pack.py 没按摘要数拼块')
                if feats is None or has_feat is None:
                    raise ValueError('--fusion-pool 开着但批次没给 feats/has_feat —— '
                                     '池化的 query 来自器官槽')
                if tower_rows is None:
                    raise ValueError('--fusion-pool 要 tower_rows（每样本视觉塔输出行数）'
                                     '—— 池化按样本切视觉塔输出，缺了就不知道从哪切')
                lens = [int(t) for t in tower_rows]
                if sum(lens) != ie.shape[0]:
                    raise ValueError(f'视觉塔输出 {ie.shape[0]} 行 != tower_rows 之和 {sum(lens)}'
                                     '（image_grid_thw 与序列不是同一批切片）')
                v32 = self.projector(feats, has_feat)
                if v32.shape[1] < self.pool.n_tok:
                    raise ValueError(f'projector 只出 {v32.shape[1]} 行，池化要 '
                                     f'{self.pool.n_tok} 个器官 query（整图槽不参与 query）')
                q_base = v32[:, :self.pool.n_tok]
                outs = []
                for b, part in enumerate(ie.split(lens)):
                    if self.fusion is not None:
                        part = self.fusion(part, feats[b], has_feat[b])
                    outs.append(self.pool(part, q_base[b]))
                ie = torch.cat(outs, 0)
            elif int(imask.sum()) * x.shape[-1] != ie.numel():
                raise ValueError(f'原生插槽 {int(imask.sum())} 个与视觉塔输出 {ie.shape[0]} 行对不上'
                                 f'（pack.py 的每片占位数应当 = t·h·w/merge²，由 grid 定）')
            if bool((imask & vis_mask).any()):
                raise ValueError('原生插槽与 RADAR 插槽相交 —— 两种占位 id 用串了')
            if self.pool is None and self.fusion is not None:
                if feats is None or has_feat is None:
                    raise ValueError('--fuse 开着但批次没给 feats/has_feat —— 融合的 K/V 无处可来')
                ie = self._fuse_native(ie, feats, has_feat, imask)
            x = x.masked_scatter(imask.unsqueeze(-1), ie.to(x.dtype))
        cnt = vis_mask.sum(dim=-1)
        if cnt.numel() and int(cnt.min()) != int(cnt.max()):
            raise ValueError(f'同一 batch 里视觉占位符个数不一致：{sorted(set(cnt.tolist()))}'
                             f'（state-only 与带视觉的样本不能混批）')
        n_vis = int(cnt.max()) if cnt.numel() else 0
        if n_vis == 0:
            return x
        if n_vis != self.n_tok:
            raise ValueError(f'序列里有 {n_vis} 个视觉占位符，projector 有 {self.n_tok} 个槽'
                             f'（{self.n_organs} 器官'
                             + ('+1 整图' if self.projector.has_whole else '') + '）')
        if feats is None or has_feat is None:
            raise ValueError('序列里有视觉占位符但没给 feats/has_feat')
        v = (v32 if v32 is not None else self.projector(feats, has_feat)).to(x.dtype)
        if self.route is not None:
            v = self.route(v, route_mask)
        if v.numel() != x.shape[0] * n_vis * x.shape[-1]:
            raise ValueError(f'projector 输出 {tuple(v.shape)} 与占位符 '
                             f'{x.shape[0]}×{n_vis}×{x.shape[-1]} 对不上')
        x = x.masked_scatter(vis_mask.unsqueeze(-1), v)
        if evs is not None:
            if int(ev_mask.sum()) * x.shape[-1] != evs.numel():
                raise ValueError(f'证据 token {tuple(evs.shape)} 与占位 {int(ev_mask.sum())} '
                                 '个对不上 —— 归属与占位不同源')
            x = x.masked_scatter(ev_mask.unsqueeze(-1), evs.to(x.dtype))
        return x

    def _positions(self, input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
        """M-RoPE 的 3D 位置：`[3,B,T]`（t/h/w），或纯文本路径的 `None`（宿主自己推）。

        为什么必须自己算：宿主 `compute_3d_position_ids` 要 `input_ids` 才算得出 M-RoPE
        （modeling_qwen3_5.py:1583-1612），而我们的前向喂的是 `inputs_embeds`
        —— 那时它**静默**退回 `attention_mask.cumsum` 的 1D 位置：图像段不再摊成 2D 网格，
        位置差全丢。传进来的 `input_ids` 是给位置函数用的，不进语言模型。

        🔴 `--fusion-pool` 时唯一可能的位置口径是 1D：序列里只剩 N 个**摘要**占位符，
        与切片网格（N 片 × 143 token）对不上；把真实 grid 交给 `get_rope_index` 会直接
        在「图像 token 数 != grid 摊开的数」上炸，或者更糟 —— 不炸但错位。摘要 token 对
        LLM 就是「带视觉证据的普通嵌入」，与 LLaVA 一类 projector 模型同构，1D 位置是
        自洽的。这里显式返回 None，而不是让调用方忘了传 grid。
        """
        if self.pool is not None or self.organ2d is not None \
                or self.evidence is not None:
            return None
        if not self.has_vision or image_grid_thw is None:
            return None
        if mm_token_type_ids is None:
            raise ValueError('宿主有多模态位置函数但批次里没 mm_token_type_ids —— '
                             'pack.py 的插槽表没出（图像位=1，其余含 RADAR 槽=0）')
        base = getattr(self.llm, 'model', self.llm)
        pos, _ = base.get_rope_index(input_ids, mm_token_type_ids,
                                     image_grid_thw=image_grid_thw,
                                     video_grid_thw=None, attention_mask=attention_mask)
        return pos

    def _restricted_scores(self, logits, answer_start, cand_ids, answer_len=None):
        """[B,T,V] logits -> [B,K] 候选分数（受限 softmax 的 logit 部分）。

        `answer_len` 给了就按它剔除「比被喂进上下文的答案更长」的候选（见 `_cand_index`）。
        """
        pos, tok, valid, cvalid, dropped = _cand_index(cand_ids, answer_start,
                                                       logits.device, answer_len)
        lg = logits.float()
        z = torch.logsumexp(lg, dim=-1)
        ar = torch.arange(logits.shape[0], device=logits.device)[:, None, None]
        sel = lg[ar, pos, tok]
        lp = sel - z[ar, pos]
        scores = (lp * valid).sum(dim=-1)
        scores = scores.masked_fill(~cvalid, float('-inf'))
        return scores, scores.argmax(dim=-1), cvalid.sum(dim=-1), dropped

    def _forward_evgrpo(self, batch, cfg):
        """armEV：获取采样 + 确定性掩码 + 答案前向，收进**一次 DDP forward**。

        为什么 fused：获取头在答案前向**之前**打分（动作改的是输入）。若在 DDP.forward
        之外单独算 acq_logits，`find_unused_parameters=True` 的图遍历从 forward 输出出发
        看不见获取头 ⇒ 把它判成未用、预先置零就绪，等 PG 的真梯度到达时二次标记报错。
        把 acquire→sample→mask→answer 收进同一次 forward，输出 dict 同时含 acq_logits
        与 scores，DDP 两条头都看见，梯度同步正确。既有 forward（evgrpo=None）逐位不变。

        `cfg = {'n_samples': G, 'n_organs': K, 'generator': torch.Generator|None}`。
        batch 是 `AcquireDataset.collate` 的 B·G 行（题主序，G 份相同全证据副本）。
        返回答案 dict 追加 `acq_logits[B,n_slots]`（带梯度）/`subsets[B,G,n_slots]`/
        `present[B,n_slots]`。loop 侧用 subsets 重建同一 masked 批喂参考策略
        （apply_acq_mask 确定性 ⇒ 与这里逐位一致）。
        """
        from .views import apply_acq_mask
        G = int(cfg['n_samples'])
        n_organs = int(cfg['n_organs'])
        gen = cfg.get('generator')
        has = batch['has_feat']
        N = int(has.shape[0])
        if G < 2 or N % G != 0:
            raise ValueError(f'evgrpo: batch {N} 行须为 G={G}(>=2) 的整数倍（题主序）')
        B = N // G
        n_slots = int(has.shape[1])
        qidx = torch.arange(0, N, G, device=has.device)
        qbatch = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                qbatch[k] = v.index_select(0, qidx)
            elif isinstance(v, list):
                qbatch[k] = [v[i] for i in qidx.tolist()]
            else:
                qbatch[k] = v
        acq_logits = self.acquire_scores(qbatch)
        present = has.index_select(0, qidx).bool()
        probs = torch.sigmoid(acq_logits).unsqueeze(1).expand(B, G, n_slots)
        u = torch.rand(probs.shape, device=probs.device, generator=gen)
        subsets = ((u < probs) & present.unsqueeze(1)).to(acq_logits.dtype)
        masked = apply_acq_mask(dict(batch), subsets.detach(), n_organs)
        out = self.forward(masked, token_ce=False)
        out['acq_logits'] = acq_logits
        out['subsets'] = subsets.detach()
        out['present'] = present
        return out

    def forward(self, batch, *, token_ce=False, evgrpo=None):
        """teacher-forced 训练路径。返回 loss 与逐候选分数（候选集 ragged，用 -inf 掩齐）。

        `token_ce=True` 走导出的 Vision-Jev 形式：labels 只覆盖 assistant 应答
        （最终符号版是裸符号，旧导出是 `Answer: …`；两者都含 `<|im_end|>`），
        loss = 这些位置上的逐 token CE，不再从 teacher-forced 答案前缀里反读候选分数。
        评估仍然只走 `score_exact()`。

        `evgrpo`（armEV，默认 None）：给了 cfg 就走获取 RL 的 fused 前向（见
        `_forward_evgrpo`）；None 时下面这条既有路径逐位不变。
        """
        if evgrpo is not None:
            return self._forward_evgrpo(batch, evgrpo)
        al = batch.get('answer_len')
        if al is None or int(al.min()) <= 0:
            raise ValueError('forward() 要求序列里带答案（answer_len > 0）；'
                             'prompt-only 的批次走 score_exact()（评估路径）')
        pv, grid = batch.get('pixel_values'), batch.get('image_grid_thw')
        pos = self._positions(batch['input_ids'], batch.get('mm_token_type_ids'),
                              grid, batch['attn_mask'])
        x = self._embed(batch['input_ids'], batch['vis_mask'],
                        batch.get('feats'), batch.get('has_feat'), pv, grid,
                        batch.get('tower_rows'), batch.get('organ_box'),
                        batch.get('slice_z'), batch.get('ev_mask'),
                        batch.get('ev_organs'), batch.get('route_mask'),
                        ev_vision_group=batch.get('ev_vision_group'),
                        pix_rows=batch.get('pix_rows'), n_img=batch.get('n_img'))
        out = self.llm(inputs_embeds=x, attention_mask=batch['attn_mask'],
                       position_ids=pos, use_cache=False)
        if token_ce:
            labels = batch['labels']
            if labels.shape != out.logits.shape[:2]:
                raise ValueError(f'labels {tuple(labels.shape)} 与 logits '
                                 f'{tuple(out.logits.shape[:2])} 对不上')
            tgt = labels[:, 1:]
            keep = tgt != -100
            n_tok = int(keep.sum())
            if n_tok == 0:
                raise ValueError('token_ce 的 batch 里没有任何答案 token（labels 全 -100）')
            b, p = keep.nonzero(as_tuple=True)
            loss = F.cross_entropy(out.logits[b, p].float(), tgt[b, p])
            return {'loss': loss, 'n_tok': n_tok, 'n_dropped': 0}
        scores, pred, n_cand, dropped = self._restricted_scores(
            out.logits, batch['answer_start'], batch['cand_ids'], al)
        target = batch['target']
        if bool((~torch.isfinite(scores).gather(1, target[:, None])).any()):
            raise RuntimeError('目标候选自己被判了 -inf —— 剔除规则把目标也剔掉了'
                               '（target 的 token 数必然 == answer_len，不该发生）')
        loss = F.cross_entropy(scores, target)
        return {'loss': loss, 'scores': scores, 'pred': pred, 'n_cand': n_cand,
                'n_dropped': len(dropped),
                'lp_target': scores.gather(1, target[:, None]).squeeze(1)}

    @torch.no_grad()
    def score_exact(self, batch, *, rows_per_chunk=8, oneshot=True):
        """评估路径：候选分数精确（见模块 docstring）。

        `batch` 是 `dataset.collate` 的 prompt-only 输出（`answer_index=None`）。
        返回 `(scores [B,K] fp32, cvalid [B,K] bool)`；无效候选为 -inf。

        `oneshot`（默认开）：本批候选**全部单 token** 时每题只推一次前缀，K 个分数取自
        同一行 logits 的第 `l-1` 位。因果注意力下该位的输出与后面接哪个候选 token 无关，
        所以逐候选复制前缀多出来的 K-1 次是重复计算（整卷 meanK=3.061 ⇒ 老口径每条臂
        多跑 117,140 次前向，且每次都重过一遍该题的视觉塔）。候选里有任何一个多 token
        （旧导出的 `Answer: …` 族，首 token 常相同、要逐位累加 `log P(tok_j | prompt +
        tok_<j)`）时自动退回逐候选路径 —— 一次前缀前向给不齐那个链。`--score-oneshot 0`
        强制走逐候选老路，用于新旧读数对拍。

        B 支下每个**行**都要**复制该样本自己的像素**：视觉塔的输出行数由 `image_grid_thw`
        定，共享一份 `pixel_values` 而只扩 `input_ids` 会让插槽数与特征行数对不上
        （`_embed` 里那条断言就是拦这个）。所以按样本的像素行界切片重拼。
        """
        input_ids = batch['input_ids']
        vis, am_in = batch['vis_mask'], batch['attn_mask']
        feats, has = batch.get('feats'), batch.get('has_feat')
        cands = batch['cand_ids']
        pv_all, grid_all = batch.get('pixel_values'), batch.get('image_grid_thw')
        mm_in = batch.get('mm_token_type_ids')
        dev = input_ids.device
        B = len(cands)
        k_max = max(len(c) for c in cands)
        out = torch.full((B, k_max), float('-inf'), device=dev)
        cvalid = torch.zeros((B, k_max), dtype=torch.bool, device=dev)
        if pv_all is not None:
            pix_rows, img_rows = batch['pix_rows'], batch['n_img']
            p0 = torch.cat([pix_rows.new_zeros(1), pix_rows.cumsum(0)])
            g0 = torch.cat([img_rows.new_zeros(1), img_rows.cumsum(0)])
        fast = bool(oneshot) and all(len(c) == 1 for cs in cands for c in cs)
        units = ([(b, list(range(len(cands[b])))) for b in range(B)] if fast
                 else [(b, [k]) for b in range(B) for k in range(len(cands[b]))])
        for s in range(0, len(units), rows_per_chunk):
            chunk = units[s:s + rows_per_chunk]
            lens = [int(am_in[b].sum()) for b, _ in chunk]
            tot = max(l + (0 if fast else len(cands[b][ks[0]]))
                      for (b, ks), l in zip(chunk, lens))
            n = len(chunk)
            ids = torch.full((n, tot), self.pad_id, dtype=torch.long, device=dev)
            am = torch.zeros((n, tot), dtype=torch.long, device=dev)
            vm = torch.zeros((n, tot), dtype=torch.bool, device=dev)
            evm = (None if batch.get('ev_mask') is None
                   else torch.zeros((n, tot), dtype=torch.bool, device=dev))
            mm = (None if mm_in is None
                  else torch.zeros((n, tot), dtype=mm_in.dtype, device=dev))
            for i, ((b, ks), l) in enumerate(zip(chunk, lens)):
                ids[i, :l] = input_ids[b, :l]
                am[i, :l] = 1
                vm[i, :l] = vis[b, :l]
                if evm is not None:
                    evm[i, :l] = batch['ev_mask'][b, :l]
                if mm is not None:
                    mm[i, :l] = mm_in[b, :l]
                if not fast:
                    c = cands[b][ks[0]]
                    ids[i, l:l + len(c)] = torch.tensor(c, dtype=torch.long, device=dev)
                    am[i, l:l + len(c)] = 1
            f = None if feats is None else torch.stack([feats[b] for b, _ in chunk])
            h = None if has is None else torch.stack([has[b] for b, _ in chunk])
            ob_in, oz_in = batch.get('organ_box'), batch.get('slice_z')
            ob = None if ob_in is None else torch.stack([ob_in[b] for b, _ in chunk])
            oz = None if oz_in is None else torch.stack([oz_in[b] for b, _ in chunk])
            evo = (None if batch.get('ev_organs') is None
                   else [batch['ev_organs'][b] for b, _ in chunk])
            rm_in = batch.get('route_mask')
            rm = None if rm_in is None else torch.stack([rm_in[b] for b, _ in chunk])
            if pv_all is None:
                pv = grid = tr = None
            else:
                pv = torch.cat([pv_all[int(p0[b]):int(p0[b + 1])] for b, _ in chunk])
                grid = torch.cat([grid_all[int(g0[b]):int(g0[b + 1])] for b, _ in chunk])
                tr = None
                if self.pool is not None:
                    tower = batch.get('tower_rows')
                    if tower is None:
                        raise ValueError('池化臂的 score_exact 缺 tower_rows —— '
                                         '数据面没记每样本视觉塔输出行数')
                    tr = tower.new_tensor([int(tower[b]) for b, _ in chunk])
                elif self.organ2d is not None:
                    tower = batch.get('tower_rows')
                    if tower is None:
                        raise ValueError('organ2d 的 score_exact 缺 tower_rows —— '
                                         '空间归属要按样本切视觉塔输出')
                    tr = tower.new_tensor([int(tower[b]) for b, _ in chunk])
                elif self.evidence is not None:
                    tower = batch.get('tower_rows')
                    if tower is None:
                        raise ValueError('organ-evidence 的 score_exact 缺 tower_rows —— '
                                         '区域归属要按样本切视觉塔输出')
                    tr = tower.new_tensor([int(tower[b]) for b, _ in chunk])
            pos = (None if grid is None else
                   self._positions(ids, mm, grid, am))
            x = self._embed(ids, vm, f, h, pv, grid, tr, ob, oz, evm, evo, rm)
            lg = self.llm(inputs_embeds=x, attention_mask=am,
                          position_ids=pos, use_cache=False).logits.float()
            z = torch.logsumexp(lg, dim=-1)
            for i, ((b, ks), l) in enumerate(zip(chunk, lens)):
                kix = torch.tensor(ks, dtype=torch.long, device=dev)
                if fast:
                    toks = torch.tensor([cands[b][k][0] for k in ks],
                                        dtype=torch.long, device=dev)
                    out[b, kix] = lg[i, l - 1, toks] - z[i, l - 1]
                else:
                    k = ks[0]
                    c = torch.tensor(cands[b][k], dtype=torch.long, device=dev)
                    j = torch.arange(len(c), device=dev)
                    out[b, k] = (lg[i, l - 1 + j, c] - z[i, l - 1 + j]).sum()
                cvalid[b, kix] = True
        return out, cvalid
