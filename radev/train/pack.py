"""Jev 记录 -> 小 LLM 的训练序列（一序列一题）+ 受限 softmax 的候选集。

计划出处（`.qoder/plans/earnest-mesa-source.md` 步骤 5、§3.2、§3.3）：

* **一序列一题，不打包**。RADAR 嵌入已缓存、重算成本为零，而打包会让第 k 题在答之前
  看见前 k-1 题的 ground truth —— `gallbladder: surgically absent ⇒ gallstones: No`
  这类共现捷径可以不看影像就走。
* **生成式读出**：答案必须是合法 Jev 应答。候选集与答案文本**只在本模块一处定义**，
  训练与评估共用（两处各写一套就是下一个「改一处忘一处」）。
* 两条机器断言（plan §6 验证清单第 4/5 条）：
  a13  答案 token 序列**单射** —— 不同候选不得编成同一串，否则受限 softmax 永远分不开
       它们。最终导出用单 token 符号（choice a..z / noul a,b / score 1..5），
       `AnswerSpace.single_token` 在装载时逐候选硬断言；内部题面或旧导出的多 token
       候选仍走序列 log-prob 求和（见 model.py）。
  a14  候选集 == 该题 criteria：choice 用 `criteria` 的 key 集合（顺序即索引序），
       score 用 `len(criteria) == config.SCORE_LEVEL_COUNT`。候选**不另抄一份病名表** ——
       从题面本身取，这就是这条断言的机器化表达。

noul 的语义约定：`out_v3` 的 `t_noul(v) = int(lab == findings.POS)`（`build/noul.py:25`）
⇒ 1 = 题面陈述成立 = `Yes`。`target_index()` 是**唯一**一处做 key -> index 映射的地方。

模板是手工拼的（不用 `apply_chat_template`）：视觉 token 块要插在 state 与题面之间，
chat 模板表达不了中间的插槽；LLaVA / Hulu-Med 也都是手工拼。切分点落在特殊 token 上，
三段独立 encode 不会引入跨边界 BPE 合并。

**B 支（宿主 = 原生多模态 Qwen3.5）序列里有两块视觉**：先 N 组
`vision_start + c×image_token + vision_end`（c 由 `slice_counts(grid)` 现算），再 36/37 个
RADAR 槽（空闲 id，见 `model.py`）。两条通道靠**不同的 token id** 分开，`mm_token_type_ids`
只有图像位是 1 —— 宿主的 M-RoPE 读这张表，读不到就退成 1D 位置。
"""
from __future__ import annotations

from dataclasses import dataclass

from .. import config

IM_START = '<|im_start|>'
IM_END = '<|im_end|>'


@dataclass(frozen=True)
class AnswerSpace:
    """一道题的答案空间。`keys` 是受限 softmax 的类名，`texts` 是送进 LLM 的答案字符串
    （与 keys 一一对应）。`ordinal` 供 metrics 选 QWK / 逐档 MAE。

    `source_texts` 是题干里展示的语义候选文本（符号模式下 `texts` 是 a/1/b，语义文本
    只用于跨卷一致性核对与报数）；`single_token=True` 是 Jev 单 token 输出的硬约束，
    `encode_space` 会在装载时逐候选断言，不允许静默退回多 token。"""
    kind: str
    keys: tuple
    texts: tuple
    ordinal: bool
    source_texts: tuple | None = None
    single_token: bool = False

    @property
    def n(self):
        return len(self.keys)


def space_for(question, ins):
    """题面 -> AnswerSpace（a14 在这里断言）。`question` 只用于报错定位。"""
    t = ins.get('type')
    if t == 'noul':
        return AnswerSpace('noul', ('No', 'Yes'), ('No', 'Yes'), False)
    if t == 'choice':
        crit = ins.get('criteria')
        if not isinstance(crit, dict) or not crit:
            raise ValueError(f'{question}: choice 题缺 criteria：{crit!r}')
        keys = tuple(crit.keys())
        if len(set(keys)) != len(keys):
            raise ValueError(f'{question}: criteria 有重复 key：{keys}')
        if 'no_abnormality' in keys and keys[-1] != 'no_abnormality':
            raise ValueError(f'{question}: no_abnormality 不在 criteria 末位：{keys}')
        return AnswerSpace('choice', keys, keys, False)
    if t == 'score':
        crit = ins.get('criteria')
        if not isinstance(crit, list) or len(crit) != config.SCORE_LEVEL_COUNT:
            raise ValueError(f'{question}: score 题 criteria 应为长度 '
                             f'{config.SCORE_LEVEL_COUNT} 的 list，实为 {crit!r}')
        if len(config.SEV_LEVELS) != len(crit):
            raise ValueError(f'config.SEV_LEVELS 有 {len(config.SEV_LEVELS)} 档，criteria 有 {len(crit)}')
        return AnswerSpace('score', tuple(f'L{k}' for k in range(len(crit))),
                           tuple(config.SEV_LEVELS), True)
    raise ValueError(f'{question}: 未知题型 {t!r}')


def target_index(space, target):
    """出厂 target -> 候选下标。`choice` 那条用 `.index()` —— target 不在候选集里时抛
    ValueError，正是 a14 的反向断言（target ⊆ 候选集）。"""
    if space.kind == 'noul':
        i = int(target['noul'])
        if i not in (0, 1):
            raise ValueError(f'noul target 应为 0/1，实为 {target!r}')
        return i
    if space.kind == 'choice':
        return space.keys.index(target['choice'])
    i = int(target['level'])
    if not 0 <= i < space.n:
        raise ValueError(f'level {i} 越界（共 {space.n} 档）')
    return i


@dataclass(frozen=True)
class EncodedSpace:
    """AnswerSpace 的 token 化结果。`single` = 全部候选都是单 token ⇒ 读出可走 gather
    快速路径；否则走序列 log-prob 求和。"""
    cand_ids: tuple
    single: bool

    @property
    def n(self):
        return len(self.cand_ids)


def _special_id(tok, token):
    i = tok.convert_tokens_to_ids(token)
    if i is None or i == tok.unk_token_id:
        raise ValueError(f'tokenizer 没有 {token}（{i!r}）—— 本模块的模板按 Qwen 系写；'
                         f'换词表要同时改 pack.py 的模板常量')
    return i


def encode_space(space, tok, lead=' '):
    """a13：逐候选编码 + 单射断言。纯函数（无缓存）—— dataset 对**唯一答案空间**建表，
    不是每题重编。

    内部题面统一带**前导空格**编码：Qwen BPE 的词首形式（Ġ 前缀）在实测里更短 ——
    `moderate`/`massive` 不带空格是 2 token，带空格是 1（`Yes`/`No`/五档其余词两边都是 1）。
    序列上等价于 prompt 末尾 `assistant\\n` 之后接一个空格再写答案。

    `lead=''` 给导出 JSON 这条通道：最终符号版的候选文本是裸符号（a/b/1..5），旧导出
    是完整的 `Answer: …`；prompt 末尾 `assistant\\n` 之后直接从答案第一个字符开始，
    不能再塞空格。
    """
    cand = []
    for text in space.texts:
        ids = tok.encode(lead + text, add_special_tokens=False)
        if not ids:
            raise ValueError(f'{space.kind}: 候选 {text!r} 编出空序列')
        cand.append(tuple(ids))
    if len(set(cand)) != len(cand):
        dup = [c for c in cand if cand.count(c) > 1]
        raise AssertionError(f'a13 违反：候选编成同一串 {sorted(set(dup))}（{space.texts}）')
    im_end = _special_id(tok, IM_END)
    for ids in cand:
        if im_end in ids:
            raise ValueError(f'a13：候选 {ids} 里混进 {IM_END}，会截断序列')
    single = all(len(c) == 1 for c in cand)
    if space.single_token and not single:
        bad = [(t, len(c)) for t, c in zip(space.texts, cand) if len(c) != 1]
        raise ValueError(f'{space.kind}: Jev 要求单 token 答案，但 {bad} 不是 —— '
                         '检查候选符号与 tokenizer 词表')
    return EncodedSpace(tuple(cand), single)


def slice_counts(grid, merge):
    """`image_grid_thw` [N,3]（patch 计数的 t/h/w）+ spatial_merge_size -> 每片占位数。

    口径是**实测**的（`analysis/probe_q35_proc.py`，2026-10-14）：512×512 -> grid
    [1,32,32] -> 256 个 `image_token`；256×384 -> [1,24,16] -> 96。即
    `t·h·w / merge²` —— h/w 已经是 patch 数，**不要再除一次 patch 尺寸**
    （除错了会得到 1 与 0 这种看着也自洽的数，探针里就踩过）。

    存在的理由：占位数必须与视觉塔输出行数**逐样本**相等，`model._embed` 才敢
    `masked_scatter`。让 dataset 从它拿到的 grid 现算，而不是抄一个常数 —— 切片分辨率
    一改常数就错，而错了是静默的少填/多填。
    """
    m2 = int(merge) ** 2
    out = []
    for t, h, w in grid.tolist():
        n = int(t) * int(h) * int(w)
        if n % m2:
            raise ValueError(f'grid ({t},{h},{w}) 的 patch 数不能被 merge²={m2} 整除 —— '
                             f'processor 输出的 grid 与宿主 config 的 merge 不同源')
        out.append(n // m2)
    return tuple(out)


def answer_hint(space):
    if space.kind == 'noul':
        return '\nAnswer with a single word: Yes or No.'
    return '\nOptions: ' + ', '.join(space.texts) + '.'


def build_example(tok, state, ins_text, space, enc, answer_index, n_vis, ph_id,
                  *, img_counts=None, img_ids=None, n_summary=None,
                  ev_organs=None, ev_ph_id=None):
    """一条 (记录, 题) -> 训练序列。`answer_index=None` ⇒ 只出 prompt（推理/打分用），
    labels 全 -100，返回值多一个 `answer_start`。

    `ph_id` 是视觉占位 token（调用方给，N_VIS 个）：序列里这些位置在 model.py 里被
    projector 输出**整体覆盖**，所以取哪个特殊 token 不影响前向，只影响可读性。

    **B 支的双插槽**（宿主是原生多模态时）：`img_counts` 是每片切片的占位数、`img_ids`
    是 `(vision_start, image_token, vision_end)` 三元组，两者都由调用方从宿主 config +
    `image_grid_thw` 给（**不在这里算** —— 这里没有 tokenizer 之外的输入尺寸信息）。
    切片块插在 RADAR 槽**之前**，两条通道各占各的 id，所以 model.py 里两次
    `masked_scatter` 互不覆盖。同时产出 `mm_token_type_ids`：宿主的 M-RoPE 只认这张表
    （图像位=1，其余=0；RADAR 槽按 0 走文本推进 —— 它是第三种模态，宿主没有对应值，
    给它 0 是**已知失真**，不是「它本来就是文本」）。

    🔴 占位符不能过 tokenizer 往返：实测 `image_token` 之类在词表尾部有空闲 id
    （248077 起 243 个，解码为空串），把它们编进文本再 encode 会被丢掉。所以序列是
    **手工拼 id 列表**，与 `slices`/processor 的产物在 dataset.py 里对齐。

    `n_summary`（`--fusion-pool`）：切片块不再是「每片 c 个占位符」，而是**一个**块里
    `n_summary` 个图像占位符 —— 视觉塔仍按真实切片出 `ΣL` 行，model 侧池化后填进这
    `n_summary` 个位置。此时必须给 `img_ids`、不能给 `img_counts`（两者定义互斥）。

    `ev_organs`（`--organ-evidence`）：题绑定器官的行号列表 —— 这些器官的占位符后面
    各插一个**证据占位符**（`ev_ph_id`），model 侧由 `OrganEvidence` 的输出覆盖。
    `ev_ph_id` 给了就恒产出 `ev_mask`（无绑定器官时全 False —— 逐题回退不断序列）。
    """
    if n_summary is not None:
        if img_counts is not None:
            raise ValueError('n_summary（池化臂）与 img_counts（逐片占位）互斥 —— '
                             '池化后序列里只有一个摘要块')
        if img_ids is None:
            raise ValueError('n_summary 需要 img_ids（宿主的 vision_start/image/end 三元组）')
        if int(n_summary) < 1:
            raise ValueError(f'n_summary={n_summary} < 1')
        if ph_id in img_ids:
            raise ValueError(f'RADAR 槽的占位 id {ph_id} 与宿主的视觉 id 撞了：{img_ids}')
    else:
        if (img_counts is None) != (img_ids is None):
            raise ValueError('img_counts 与 img_ids 必须同时给或同时不给 —— '
                             '半条原生视觉通路在 model.py 里会静默错位')
        if img_counts is not None and ph_id in img_ids:
            raise ValueError(f'RADAR 槽的占位 id {ph_id} 与宿主的视觉 id 撞了：{img_ids}')
    head = tok.encode(f'{IM_START}user\n{state}\n\n', add_special_tokens=False)
    tail = tok.encode(f'{ins_text}{answer_hint(space)}{IM_END}\n{IM_START}assistant\n',
                      add_special_tokens=False)
    vs, it, ve = img_ids if img_ids is not None else ((), (), ())
    ev = _norm_ev(ev_organs, ev_ph_id, n_vis, ph_id, img_ids)
    vis_block = []
    blk_vis, blk_ev = [], []
    for k in range(n_vis):
        vis_block.append(ph_id)
        blk_vis.append(True)
        blk_ev.append(False)
        if k in ev:
            vis_block.append(ev_ph_id)
            blk_vis.append(False)
            blk_ev.append(True)
    img_block = []
    mm_img = []
    if n_summary is not None:
        img_block = [vs] + [it] * int(n_summary) + [ve]
        mm_img = [0] + [1] * int(n_summary) + [0]
    elif img_counts is not None:
        for c in img_counts:
            if c < 1:
                raise ValueError(f'切片占位数 {c} < 1：grid 算出来的每片 token 数不可能非正')
            img_block += [vs] + [it] * c + [ve]
            mm_img += [0] + [1] * c + [0]
    pre = img_block + vis_block
    pose = len(head) + len(pre) + len(tail)
    input_ids = head + pre + tail
    labels = [-100] * pose
    vis_mask = ([False] * (len(head) + len(img_block)) + blk_vis
                + [False] * len(tail))
    ev_mask = ([False] * (len(head) + len(img_block)) + blk_ev
               + [False] * len(tail))
    mm = None if (img_counts is None and n_summary is None) else (
        [0] * len(head) + mm_img + [0] * (len(vis_block) + len(tail)))
    if answer_index is None:
        answer_len = 0
    else:
        ans = list(enc.cand_ids[answer_index])
        end = [_special_id(tok, IM_END)]
        input_ids = input_ids + ans + end
        labels = labels + ans + end
        vis_mask = vis_mask + [False] * (len(ans) + 1)
        ev_mask = ev_mask + [False] * (len(ans) + 1)
        if mm is not None:
            mm = mm + [0] * (len(ans) + 1)
        answer_len = len(ans)
    for name, vec in (('labels', labels), ('vis_mask', vis_mask),
                      ('mm_token_type_ids', mm), ('ev_mask', ev_mask)):
        if vec is not None and len(vec) != len(input_ids):
            raise AssertionError(f'a18 违反：{name} 长 {len(vec)} != input_ids 长 '
                                 f'{len(input_ids)}（n_vis={n_vis} img_counts={img_counts}）')
    out = {'input_ids': input_ids, 'labels': labels, 'vis_mask': vis_mask,
           'answer_start': pose, 'answer_len': answer_len}
    if ev_ph_id is not None:
        out['ev_mask'] = ev_mask
        out['ev_organs'] = list(ev)
    if mm is not None:
        out['mm_token_type_ids'] = mm
    return out


def _norm_ev(ev_organs, ev_ph_id, n_vis, ph_id, img_ids):
    """证据器官行号 -> 排序去重的 set。`ev_ph_id` 给了才算数（None = 臂没开）。"""
    if ev_ph_id is None:
        if ev_organs:
            raise ValueError('给了 ev_organs 却没给 ev_ph_id —— 证据占位符没有 id 可用')
        return set()
    if ev_ph_id == ph_id or (img_ids is not None and ev_ph_id in img_ids):
        raise ValueError(f'证据占位 id {ev_ph_id} 与器官槽/图像插槽撞了 —— 必须是空闲 id')
    if not ev_organs:
        return set()
    ev = {int(o) for o in ev_organs}
    if min(ev) < 0 or max(ev) >= n_vis:
        raise ValueError(f'ev_organs {sorted(ev)} 越界（器官行号 0..{n_vis - 1}）')
    return ev


def build_export_example(tok, prompt_text, cand_ids, answer_index, n_vis, ph_id,
                         *, img_counts=None, img_ids=None, n_summary=None,
                         ev_organs=None, ev_ph_id=None):
    """导出的 train_val.json / test.json 一条记录 -> 训练序列（Vision-Jev 那条形式）。

    与 `build_example` 的差别只有题面来源：user 内容逐字取自导出文件
    （`<image>\\n` 之后的整段），不再用 `state + instructions + answer_hint` 另拼一份。
    视觉槽插在 `<image>` 原位；`prompt_text` 前面补一个换行，因为导出 user 内容是
    `<image>\\n<题面>`。答案段是 assistant 的应答（最终符号版是裸符号，旧导出是
    `Answer: …`），labels 只覆盖这段 + `<|im_end|>` —— 即 `_answer_only_loss`。
    """
    if n_summary is not None:
        if img_counts is not None:
            raise ValueError('n_summary（池化臂）与 img_counts（逐片占位）互斥')
        if img_ids is None:
            raise ValueError('n_summary 需要 img_ids')
        if int(n_summary) < 1:
            raise ValueError(f'n_summary={n_summary} < 1')
        if ph_id in img_ids:
            raise ValueError(f'RADAR 槽的占位 id {ph_id} 与宿主的视觉 id 撞了：{img_ids}')
    else:
        if (img_counts is None) != (img_ids is None):
            raise ValueError('img_counts 与 img_ids 必须同时给或同时不给')
        if img_counts is not None and ph_id in img_ids:
            raise ValueError(f'RADAR 槽的占位 id {ph_id} 与宿主的视觉 id 撞了：{img_ids}')
    head = tok.encode(f'{IM_START}user\n', add_special_tokens=False)
    tail = tok.encode('\n' + prompt_text + f'{IM_END}\n{IM_START}assistant\n',
                      add_special_tokens=False)
    vs, it, ve = img_ids if img_ids is not None else ((), (), ())
    ev = _norm_ev(ev_organs, ev_ph_id, n_vis, ph_id, img_ids)
    vis_block = []
    blk_vis, blk_ev = [], []
    for k in range(n_vis):
        vis_block.append(ph_id)
        blk_vis.append(True)
        blk_ev.append(False)
        if k in ev:
            vis_block.append(ev_ph_id)
            blk_vis.append(False)
            blk_ev.append(True)
    img_block, mm_img = [], []
    if n_summary is not None:
        img_block = [vs] + [it] * int(n_summary) + [ve]
        mm_img = [0] + [1] * int(n_summary) + [0]
    elif img_counts is not None:
        for c in img_counts:
            if c < 1:
                raise ValueError(f'切片占位数 {c} < 1')
            img_block += [vs] + [it] * c + [ve]
            mm_img += [0] + [1] * c + [0]
    pre = img_block + vis_block
    pose = len(head) + len(pre) + len(tail)
    input_ids = head + pre + tail
    labels = [-100] * pose
    vis_mask = ([False] * (len(head) + len(img_block)) + blk_vis
                + [False] * len(tail))
    ev_mask = ([False] * (len(head) + len(img_block)) + blk_ev
               + [False] * len(tail))
    mm = None if (img_counts is None and n_summary is None) else (
        [0] * len(head) + mm_img + [0] * (len(vis_block) + len(tail)))
    if answer_index is None:
        answer_len = 0
    else:
        ans = list(cand_ids[answer_index])
        end = [_special_id(tok, IM_END)]
        input_ids = input_ids + ans + end
        labels = labels + ans + end
        vis_mask = vis_mask + [False] * (len(ans) + 1)
        ev_mask = ev_mask + [False] * (len(ans) + 1)
        if mm is not None:
            mm = mm + [0] * (len(ans) + 1)
        answer_len = len(ans)
    for name, vec in (('labels', labels), ('vis_mask', vis_mask), ('mm_token_type_ids', mm),
                      ('ev_mask', ev_mask)):
        if vec is not None and len(vec) != len(input_ids):
            raise AssertionError(f'a18 违反：{name} 长 {len(vec)} != input_ids 长 '
                                 f'{len(input_ids)}（n_vis={n_vis} img_counts={img_counts}）')
    out = {'input_ids': input_ids, 'labels': labels, 'vis_mask': vis_mask,
           'answer_start': pose, 'answer_len': answer_len}
    if ev_ph_id is not None:
        out['ev_mask'] = ev_mask
        out['ev_organs'] = list(ev)
    if mm is not None:
        out['mm_token_type_ids'] = mm
    return out
