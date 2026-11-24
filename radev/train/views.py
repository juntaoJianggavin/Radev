"""RACD 阶段 1 的证据视图：把 RADAR 器官 mask 变成「同一道题的反事实证据」。

阶段 1 的潜变量不是文本 rationale（我们是直接打分器，没有 rationale），而是**证据
视图 e**：同一道题、同一段序列、同一个 target，只改「哪些器官的 RADAR 特征在场」。
这样 RLCD 的两阶段结构才成立 —— `u(·|x,e)` 是逐视图的读出口径，proper score 逐视图
施加（Prop 2 的 per-rationale 目标），比较项施加在视图**之间**（RAC 的成对 hinge）。

mask 的注入点是 `projector.OrganProjector.forward` 的
`torch.where(has_feat, x, missing)`：把某个器官的 `has_feat` 翻成 False，它就换成
**可学的 missing 向量** —— 这是模型在 SFT 里见过的分布内状态（一个腹部 CT 本来就
不会同时显示 36 个器官），不是往输入里加噪声。器官 id 嵌入照加，所以「缺这个器官」
与「有这个器官但特征弱」仍是两种不同输入。

三种视图（`plan_views`）：

* `full`：原样。**恒占槽 0** —— 比较项都以它为基准。
* `target_mask`：把本题**绑定的全部器官**（`organ_crop_map.tsv` 的 organs 列 ∩ 本卷
  在场器官）一次全屏蔽。整组屏而不是逐个屏，是因为 score 题的证据常是多器官的
  （肝脂肪浸润要看肝**和**脾）：只屏一半，答案仍可导出，「置信度该下降」这条约束
  就成了错的。
* `distractor_mask`：随机屏**一个**在场但不绑定的器官。单个器官是最小反事实 ——
  「这条证据与本题无关」这句话只对单个器官成立，屏掉一堆无关器官是另一种干预
  （「只剩绑定器官」），语义不同。

`route_mask` 在所有视图里**保持不变**（`--organ-route` 的题绑定门控照旧上调绑定槽）：
被屏蔽的器官仍被路由指着，只是指过去发现证据不在了。这正是要的因果对比 —— 变的只有
证据在场性，不是「模型该看哪儿」。

🔴 本模块不改 SFT 路径：`MultiViewDataset` 是 `dataset.JevDataset` 的**外挂包装**，
每个 index 只调一次 base 的 `__getitem__`（题面/序列/像素全复用），M 个视图之间只差
一份 clone 过的 `has_feat`。属性访问经 `__getattr__` 透传，`_check_rows` /
`_dump_slice_stats` / `describe` 都照旧工作。
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import os
import random

import torch
from torch.utils.data import Sampler

from . import radar_encoder as RE

VIEW_FULL = 'full'
VIEW_TARGET = 'target_mask'
VIEW_DISTRACTOR = 'distractor_mask'
VIEW_BLIND = 'blind'

NO_GROUP = '__unbound__'


def load_organ_bindings(path):
    """`organ_crop_map.tsv` -> ({题名: (组名, 绑定器官行号 tuple)}, sha256, 行数)。

    与 `loop._load_organ_map` 读同一张表、同一套校验（表头 / 器官名 ∈ ORGANS / 题名
    不重复），多带一列 `group` —— 分层采样按组配额，路由/裁剪臂不需要组名所以那边没留。
    这里返回**行号**（`RE.ORGANS.index`）而不是器官名：mask 直接作用在 `has_feat` 的
    行上，名字到行号的换算只该做一次。
    """
    if not os.path.isfile(path):
        raise SystemExit(f'缺器官表 {path} —— RACD 的视图与分层采样都靠它；'
                         '先生成：`analysis/organ_crop_map.py`')
    out = {}
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        if head[:3] != ['question', 'group', 'organs']:
            raise SystemExit(f'{path} 表头 {head[:3]} != [question, group, organs] —— '
                             '不是 analysis/organ_crop_map.py 的产物？')
        for ln, line in enumerate(f, 2):
            d = line.rstrip('\n').split('\t')
            q, grp = d[0], d[1]
            orgs = d[2].split() if len(d) > 2 else []
            bad = [o for o in orgs if o not in RE.ORGANS]
            if bad:
                raise SystemExit(f'{path}:{ln} 未知器官 {bad}（对照 radar_encoder.ORGANS）')
            if q in out:
                raise SystemExit(f'{path}:{ln} 题名 {q} 重复')
            out[q] = (grp, tuple(sorted(RE.ORGANS.index(o) for o in orgs)))
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return out, sha, len(out)


def plan_views(bound, present, n_views, rng, whole_row=None, blind=False):
    """-> `[(kind, mask_rows), ...]`，长度恰为 `n_views`，槽 0 恒 `full`。

    `bound` = 本题绑定器官行号；`present` = 本卷 `has_feat=True` 的槽行号（`whole_row`
    给了才可能含整图槽，见下）。抽不出目标/干扰时该槽退回 `full`：宁可少一条比较项，也不要
    造一个语义不明的视图。

    `whole_row`（`--racd-mask-whole`，默认 `None` = 老口径：整图槽豁免）：把**整图全局槽**
    并进 target_mask。豁免原本是错的，实测证据是 racd1 的 `gate_frac`：0.0431（step 20）→
    0.0194（step 2000）——「遮掉绑定器官后答对率真的下降」只在 2~4% 的行上成立，λ_corr=0.3
    被稀释成 ≈0.012，corr 轴基本空转。原因在通路：organ-evidence 臂的切片 token **不进
    序列**（model.py 对 `imask.sum()==0` 有硬断言），视觉塔到 LLM 只剩两条路 —— 绑定器官的
    证据 token（已被 `organ_box=-1` 拿掉）和 37 个 RADAR 槽。绑定器官槽换成 missing 之后
    **整图槽仍带着 `whole_feat`**，那是全卷的全局摘要，病灶大概率还在里面，答案照样读得出。
    所以「豁免整图槽」不是保守，是给反事实留了一条泄漏通路。
    （整图槽的**证据** token 不必管：dataset.py 是 `if organs: … elif …`，第 37 槽只在
    「无绑定器官」的题上进 `ev_organs`，而 target_mask 只在有绑定的题上存在 —— 那条 K/V
    分支在遮蔽视图里根本不跑。）

    🔴 整图槽只进 target_mask，**绝不单独当 distractor**：distractor 的语义是「这条证据与本题
    无关，拿掉它置信度不该变」，而全局摘要对任何题都不是无关证据，拿掉它真会改后验 ——
    对它施加不变性约束是错的约束（与下面无绑定题同一条理由）。

    `blind`（`--racd-blind`，默认关 = 既有臂逐位不变）：把**最后一槽**换成盲视图 ——
    全部在场槽（含整图槽）一次屏掉。它与前两种视图的语义根本不同：

    * target/distractor 问的是「拿掉**这一条**证据，信心该不该掉」，需要一个绑定集来定义
      「这一条」是哪条；blind 问的是「**什么证据都没有**时你还剩多少信心」，那是语言先验的
      读数，对任何题都有定义 ⇒ 无绑定题（triage）也有盲视图，不像 target/distractor 那样
      只能退回 full。
    * 所以 blind 不是第三个不变性目标，而是一条**信心下降的下界**：`c_full − c_blind ≥ m_blind`。
      它抓的正是本项目那个 0.7605 的语言先验地板 —— 「不看图按经验众数就能答对」的那批题，
      在盲视图下信心照样高，`c_full − c_blind ≈ 0` ⇒ 违反下界 ⇒ 这份信心不是证据支撑的。
    * 🔴 盲视图恒占**最后一槽**，前 `n_views−1` 槽的排布与 RNG 抽样序列和不开 blind 时
      **逐位相同**（`rng.choice` 只在填 distractor 时被调，调用次数由 `n_fill` 定）⇒
      `VIEWS=4 + blind` 的槽 0/1/2 就是 `VIEWS=3` 那三个视图，blind 是**纯增量**的第四槽。
    * 🔴 blind 要求 `mask_whole=1`（或这条臂序列里压根没有第 37 槽）：否则整图槽仍带
      `whole_feat`，「盲」是假的 —— 那正是上面 `gate_frac` 0.043→0.019 那条泄漏通路。
      这条闸在 `MultiViewDataset.__init__` 里当场炸，不静默降级。

    `bound` 为空（题名不在 organ_crop_map.tsv 里，如 triage_acuity）→ **三槽全 full**。
    「干扰器官」是相对绑定集定义的，没有绑定集就没有干扰可言；而且那种题的证据本来就
    弥散（分诊紧急度看的是全图findings），随便屏一个器官是真会改后验的 —— 对它施加
    不变性约束是**错的**约束。它们只贡献 task 项，等于给非器官题留一条复现锚。
    （开了 `blind` 时这条只对前 `n_views−1` 槽成立：盲视图不依赖绑定集，无绑定题照给。）
    """
    if n_views < 2:
        raise ValueError(f'n_views={n_views} < 2 —— 只有一个视图就没有可比的两端')
    if blind and n_views < 3:
        raise ValueError(f'n_views={n_views} + 盲视图 < 3 —— 盲视图恒占最后一槽，'
                         '剩下的至少要是 full 与 target 两端')
    n_fill = n_views - 1 if blind else n_views
    if not bound:
        plans = [(VIEW_FULL, ())] * n_fill
    else:
        plans = [(VIEW_FULL, ())]
        b, p = set(bound), set(present)
        tgt = sorted(b & p)
        if whole_row is not None and tgt and whole_row in p:
            tgt.append(whole_row)
        plans.append((VIEW_TARGET, tuple(tgt)) if tgt else (VIEW_FULL, ()))
        dis = sorted(p - b - ({whole_row} if whole_row is not None else frozenset()))
        used = set()
        while len(plans) < n_fill:
            rest = [o for o in dis if o not in used]
            if not rest:
                plans.append((VIEW_FULL, ()))
                continue
            o = rng.choice(rest)
            used.add(o)
            plans.append((VIEW_DISTRACTOR, (o,)))
    if blind:
        rows = tuple(sorted(present))
        plans.append((VIEW_BLIND, rows) if rows else (VIEW_FULL, ()))
    return plans


class MultiViewDataset:
    """把 base dataset 的每个 index 展开成 `n_views` 个视图样本（**组主序**）。

    `__getitem__` 返回一个长度 M 的 dict 列表，`collate` 把 B 个组摊平成 B·M 行后交给
    base 的 collate，再挂上 `view_kind`/`view_group`/`n_views`。摊平序 = (组序, 视图序)，
    所以行号 `r` 对应组 `r // M`、视图槽 `r % M` —— `racd_loss` 就按这个约定 reshape。

    🔴 `base[i]` 给出的 `feats`/`has_feat` **是 `_feat_cache`（LRU-4）里那份对象本身**，
    不是副本。就地改一个 bit 会污染缓存，同一 study 的后续题会拿到被屏蔽过的特征 ——
    这种错不报错，只让读数变假。所以屏蔽视图一律 `has.clone()` 后再写。
    `feats` 不用动：projector 用 `has_feat` 整行换 `missing`，被掩行的特征值不参与前向。

    🔴 organ-evidence 臂同理要 clone `organ_box` 再写：屏蔽一个器官时，除了 `has_feat=False`
    把证据 token 的 query 换成 `missing`，还要把 `organ_box` 那一行整行设 -1 拿掉 box 区域
    的 K/V（那条路走几何、不受 has_feat 控制），否则反事实只做了「一半」。见 `__getitem__`。
    🔴 `organ_box` 只有 `n_organs`（36）行而 `has_feat` 有 `n_rows`（37）行 —— 整图槽没有
    box。开了 `mask_whole` 之后 mask_rows 会含 36，写 `organ_box` 前必须先滤掉越界行号，
    否则 IndexError。见 `__getitem__`。
    """

    def __init__(self, base, bindings, *, n_views=3, seed=0, mask_whole=False, blind=False,
                 prior=None):
        self.base = base
        self.bindings = bindings
        self.n_views = int(n_views)
        self.seed = int(seed)
        self.blind = bool(blind)
        self.prior = prior
        self._epoch = 0
        self.view_stats = collections.Counter()
        n_organs = int(base.n_organs)
        self.group_of = [bindings.get(ex[2], (NO_GROUP, ()))[0] for ex in base.examples]
        self.study_of = [ex[1] for ex in base.examples]
        qspace = getattr(base, '_json_qspace', None)
        self.kind_of = ([qspace[ex[2]][0].kind for ex in base.examples]
                        if qspace is not None else None)
        self.mask_whole = bool(mask_whole)
        n_vis = int(getattr(base, 'n_vis', 0))
        self.whole_row = n_organs if (self.mask_whole and n_vis == n_organs + 1) else None
        self.n_organs_lim = n_vis if self.whole_row is not None else n_organs
        if self.blind and self.whole_row is None and n_vis == n_organs + 1:
            raise ValueError('--racd-blind 要 --racd-mask-whole 一起开：这条臂序列里有第 37 个'
                             f'整图槽（n_vis={n_vis} == n_organs+1），不遮它的话盲视图仍带 '
                             'whole_feat，c_full − c_blind 量到的是泄漏而不是先验')

    def __getattr__(self, k):
        return getattr(self.base, k)

    def __len__(self):
        return len(self.base)

    def set_view_epoch(self, epoch):
        """换 epoch 换干扰器官的抽样（`GroupStratifiedSampler.set_epoch` 会调它）。"""
        self._epoch = int(epoch)

    def describe(self):
        d = dict(self.base.describe())
        d['multiview'] = {'n_views': self.n_views, 'seed': self.seed,
                          'mask_whole': self.mask_whole, 'whole_row': self.whole_row,
                          'blind': self.blind,
                          'n_groups': len(set(self.group_of)),
                          'kind_rows': (dict(collections.Counter(self.kind_of))
                                        if self.kind_of is not None else None),
                          'view_stats': dict(self.view_stats)}
        return d

    @property
    def organ_stats(self):
        d = dict(self.base.organ_stats)
        d.update({f'view_{k}': v for k, v in self.view_stats.items()})
        return d

    def __getitem__(self, i):
        ex = self.base.examples[i]
        full = self.base[i]
        has = full['has_feat']
        n_org = min(self.n_organs_lim, int(has.shape[0]))
        present = [o for o in range(n_org) if bool(has[o])]
        grp, bound = self.bindings.get(ex[2], (NO_GROUP, ()))
        rng = random.Random(f'{self.seed}:{self._epoch}:{i}')
        plans = plan_views(bound, present, self.n_views, rng, whole_row=self.whole_row,
                           blind=self.blind)
        self.view_stats['groups'] += 1
        if not bound:
            self.view_stats['no_binding'] += 1
        out = []
        for kind, rows in plans:
            self.view_stats[kind] += 1
            d = dict(full)
            if rows:
                r = list(rows)
                h = has.clone()
                h[r] = False
                d['has_feat'] = h
                if 'organ_box' in full:
                    ob = full['organ_box'].clone()
                    rb = [x for x in r if x < ob.shape[0]]
                    if rb:
                        ob[rb] = -1
                        d['organ_box'] = ob
            d['view_kind'] = kind
            d['view_group'] = grp
            out.append(d)
        n_fill = len(plans) - 1 if self.blind else len(plans)
        if bound:
            if plans[1][0] != VIEW_TARGET:
                self.view_stats['target_absent'] += 1
            elif self.whole_row is not None and self.whole_row in plans[1][1]:
                self.view_stats['target_whole'] += 1
            for slot in range(2, n_fill):
                if plans[slot][0] != VIEW_DISTRACTOR:
                    self.view_stats['distractor_absent'] += 1
        if self.blind and plans[-1][0] != VIEW_BLIND:
            self.view_stats['blind_absent'] += 1
        return out

    def collate(self, groups):
        flat = [d for g in groups for d in g]
        out = self.base.collate(flat)
        out['view_kind'] = [d['view_kind'] for d in flat]
        out['view_group'] = [d['view_group'] for d in flat]
        out['n_views'] = self.n_views
        if self.prior is not None:
            M = int(self.n_views)
            pi = [self.prior.get(d['question'], float('nan')) for d in flat]
            for g in range(len(pi) // M):
                v = pi[g * M]
                for m in range(1, M):
                    w = pi[g * M + m]
                    if v != w and not (v != v and w != w):
                        raise SystemExit(f'collate: 组 {g} 的 π 不一致（{v} vs {w}）'
                                         '—— 组主序被破坏了，逐行查表也救不了')
            out['prior'] = torch.tensor(pi, dtype=torch.float32).view(-1, M)[:, 0].contiguous()
        return out


class KeepBoundDataset:
    """**评估专用**的证据剪枝：只留本题绑定器官槽（可选带整图槽），其余全屏蔽。

    与 `MultiViewDataset` 同一套注入点（`has_feat=False` + `organ_box=-1`，序列结构/
    packing 全不动），但方向**相反**：RACD 屏掉的是绑定器官（造反事实，问「拿掉证据
    答案会不会变坏」），这里屏掉的是**非**绑定器官，问「视觉信息只走绑定器官那条路，
    够不够答题」。

    🔴 **这不是省 token 的读数，别当效率报**。`has_feat=False` 只是让 projector 把那一槽
    换成可学的 `missing` 向量（projector.py:78），槽还在序列里、占位符还是一个 →
    **序列长度与视觉 token 数一位都没变**（每器官槽恒 1 个占位符，见 pack.py:238-247；
    而 organ-evidence 臂的切片 token 根本不进序列，model.py:527 对 `imask.sum()==0` 有
    硬断言）。所以这条臂量的是**信息充分性**，不是预算削减。

    🔴 **`bound_whole` 的「没掉」不可解释**。整图第 37 槽带的是 `whole_feat`——全卷全局
    摘要，病灶大概率还在里面。这正是把 racd1 的 `gate_frac` 从 0.0431 压到 0.0194、让
    λ_corr 空转的同一条泄漏通路（见上面 `plan_views` 的 `whole_row` 段）。留着它再读到
    「准确率没掉」，分不清是「绑定器官够了」还是「答案从整图槽漏进来的」。**可解释的
    口径是 `bound`**（连第 37 槽一起屏）；`bound_whole` 只当它的对照上沿。

    🔴 只挂评估路径（loop.py 的 `prompt_only=True` 分支）。挂到训练路径上会改掉
    SFT/RACD 臂的证据面，那些臂的读数当场失去可比性。

    🔴 无绑定的题（`NO_GROUP`，实测 15.67%）与**绑定器官一个都不在场**的题，都**一个槽
    不屏**：前者没有绑定集就没有「非绑定器官」可言（那种题的证据本来就弥散，分诊紧急度看
    的是全图 findings，随便屏是真会改后验的）；后者「只留绑定器官」等于「什么都不留」，
    造出来的是零证据行 —— 它测的是「不看图靠语言先验能答对多少」（三级地板的事），不是
    「剪枝掉多少准确率」，两种读数混在一列里就都不可解释了。两类各自单独计数，别让它们
    把剪枝读数稀释成「什么都没掉」。与 `plan_views` 的 `if tgt else (VIEW_FULL, ())`
    同一条家法。

    绑定统计在 `__init__` 里一次算完（不依赖 `has_feat`，因而与 DataLoader 的 worker
    数无关、可复算）；实际剪掉几个槽要看这一卷哪些器官在场，只能在 `__getitem__` 里累
    加，`num_workers>0` 时那份计数是 worker 本地的 —— 所以它只当诊断，**读数一律以
    `bound_of` 那份为准**。
    """

    def __init__(self, base, bindings, *, keep_whole=True):
        n_organs = int(base.n_organs)
        n_vis = int(getattr(base, 'n_vis', 0))
        if n_vis == 0:
            raise SystemExit(f'--eval-keep-bound 要 RADAR 器官槽进序列，但这条臂 n_vis=0'
                             f'（n_organs={n_organs}）—— 没有槽可剪')
        self.base = base
        self.bindings = bindings
        self.keep_whole = bool(keep_whole)
        self.whole_row = n_organs if (n_vis == n_organs + 1) else None
        self.n_lim = n_vis
        self.stats = collections.Counter()
        self.bound_of = [bindings.get(ex[2], (NO_GROUP, ()))[1] for ex in base.examples]
        for b in self.bound_of:
            self.stats['rows'] += 1
            if not b:
                self.stats['no_binding'] += 1
            else:
                self.stats['n_bound_slots'] += len(set(b) & set(range(n_organs)))
                self.stats['bound_rows'] += 1

    def __getattr__(self, k):
        return getattr(self.base, k)

    def __len__(self):
        return len(self.base)

    def describe(self):
        d = dict(self.base.describe())
        s = self.stats
        nb = max(1, s['bound_rows'])
        d['keep_bound'] = {'keep_whole': self.keep_whole, 'whole_row': self.whole_row,
                           'n_lim': self.n_lim, 'rows': s['rows'],
                           'no_binding': s['no_binding'],
                           'mean_bound_slots': round(s['n_bound_slots'] / nb, 3),
                           'mean_pruned_slots': round(s['n_pruned'] / max(
                               1, s['pruned']), 3),
                           'note': 'no_binding（15.67%）与 bound_absent 的题一个槽都不剪'
                                   '（见 __getitem__）；mean_pruned_slots / bound_absent / '
                                   'pruned 是 worker 本地计数，num_workers>0 时只当诊断'}
        return d

    @property
    def organ_stats(self):
        d = dict(self.base.organ_stats)
        d.update({f'keep_bound_{k}': v for k, v in self.stats.items()})
        return d

    def __getitem__(self, i):
        d = self.base[i]
        bound = self.bound_of[i]
        if not bound:
            return d
        has = d['has_feat']
        n_org = min(self.n_lim, int(has.shape[0]))
        present = [o for o in range(n_org) if bool(has[o])]
        if not any(o in bound for o in present):
            self.stats['bound_absent'] += 1
            return d
        keep = set(bound)
        if self.whole_row is not None and self.keep_whole:
            keep.add(self.whole_row)
        rows = [o for o in present if o not in keep]
        if not rows:
            self.stats['nothing_to_prune'] += 1
            return d
        self.stats['pruned'] += 1
        self.stats['n_pruned'] += len(rows)
        d = dict(d)
        h = has.clone()
        h[rows] = False
        d['has_feat'] = h
        if 'organ_box' in d:
            ob = d['organ_box'].clone()
            rb = [x for x in rows if x < ob.shape[0]]
            if rb:
                ob[rb] = -1
                d['organ_box'] = ob
        return d


def apply_acq_mask(batch, subsets, n_organs):
    """armEV：把采到的获取子集**确定性**施加到 batch 的 has_feat/organ_box。

    `batch` 是 B·G 行（题主序，行 `b·G+g` = 题 b 的第 g 个子集，`AcquireDataset.collate`
    的约定）；`subsets [B,G,n_slots]` 0/1（n_slots=n_organs+1，末位整图）。对每行：
    * 在场但**未获取**的器官 -> `has_feat=False`（projector 换 missing）+ `organ_box=-1`
      （证据 K/V 的几何路也拿掉）—— 与 MultiViewDataset 同一条注入点；
    * **未获取**的整图 -> `has_feat[whole]=False`（整图无 box，has_feat 是唯一杠杆，
      model._evidence_embeds 的 acquire 分支据此把整图证据退回 missing）。
    序列结构/ev_organs/packing 全不动 —— G 个子集序列逐位相同，只有证据在场性不同。
    🔴 clone 再写：batch 的 has_feat/organ_box 源自 `_feat_cache`，就地改会污染缓存。
    """
    has = batch['has_feat'].clone()
    ob = batch.get('organ_box')
    if ob is not None:
        ob = ob.clone()
    B, G, n_slots = subsets.shape
    if has.shape[0] != B * G:
        raise ValueError(f'apply_acq_mask: batch {has.shape[0]} 行 != B·G={B * G}')
    s = subsets.reshape(B * G, n_slots).to(has.device).bool()
    org_acq = s[:, :n_organs]
    mask_org = has[:, :n_organs] & ~org_acq
    has[:, :n_organs] = has[:, :n_organs] & org_acq
    if ob is not None:
        ob[mask_org] = -1
    if n_slots > n_organs:
        has[:, n_organs] = has[:, n_organs] & s[:, n_organs]
    batch['has_feat'] = has
    if ob is not None:
        batch['organ_box'] = ob
    batch['ev_vision_group'] = int(G)
    return batch


class AcquireDataset:
    """armEV：把 base 的每个 index 展开成 G 个**相同**的全证据副本（题主序）。

    与 `MultiViewDataset` 的分工：那边在 dataset 侧就造好 M 个**不同**的反事实视图；
    这里 G 个副本在 dataset 侧完全相同（都是 full 证据），获取掩码在 loop 侧 Bernoulli
    采样后由 `apply_acq_mask` 施加 —— 动作改的是输入，副本只是给 B·G 行前向占位。
    collate 摊平成 B·G 行，行 `b·G+g` = 题 b 的第 g 个子集（evgrpo_loss 按此 reshape）。

    sampler 属性（group_of/study_of/kind_of）与 MultiViewDataset 同口径 —— armEV 复用
    `--grpo-balance group` 的 GroupStratifiedSampler，它要按 (组, study) 归堆。
    """

    def __init__(self, base, bindings, *, n_samples=8):
        self.base = base
        self.bindings = bindings
        self.n_samples = int(n_samples)
        if self.n_samples < 2:
            raise ValueError(f'n_samples={self.n_samples} < 2 —— 组内归一化至少要两个子集')
        self.group_of = [bindings.get(ex[2], (NO_GROUP, ()))[0] for ex in base.examples]
        self.study_of = [ex[1] for ex in base.examples]
        qspace = getattr(base, '_json_qspace', None)
        self.kind_of = ([qspace[ex[2]][0].kind for ex in base.examples]
                        if qspace is not None else None)

    def __getattr__(self, k):
        return getattr(self.base, k)

    def __len__(self):
        return len(self.base)

    def set_view_epoch(self, epoch):
        pass

    def describe(self):
        d = dict(self.base.describe())
        d['acquire'] = {'n_samples': self.n_samples,
                        'n_groups': len(set(self.group_of)),
                        'kind_rows': (dict(collections.Counter(self.kind_of))
                                      if self.kind_of is not None else None)}
        return d

    @property
    def organ_stats(self):
        return self.base.organ_stats

    def __getitem__(self, i):
        d = self.base[i]
        return [dict(d) for _ in range(self.n_samples)]

    def collate(self, groups):
        flat = [d for g in groups for d in g]
        return self.base.collate(flat)


class GroupStratifiedSampler(Sampler):
    """按题绑定器官组做**温度分层 + run 级交错**的采样（RACD 阶段 1）。

    为什么不能直接用 `StudyShuffledSampler`：均匀采样下最小组（Uterus and ovaries，
    78/223286 行）在 2000 步 × accum8 × 4 卡 = 64000 题里只摊到 ~22 次 target_mask ——
    那一组的 Δ_target 根本测不出来。温度 `p_g ∝ n_g^tau`（tau=0.5）把它抬到 ~103，
    再叠一个 `min_frac` 地板抬到 ~320，`max_repeat` 封顶防止 78 行被复读上百遍。

    三处实现约定：

    1. **run = 同一 study 在同一组里的全部题**。按 run 发牌而不是按题发牌，是为了保住
       `_feat_cache`（LRU-4）的局部性：一个 run 内的题共用一份 npz，逐题随机抽会把
       每次取样都变成一次 NAS 读。
    2. **run 级交错**（不是「各组发完再拼」）：发牌序列按组配额轮转生成，所以**任意
       前缀**的组构成都接近全局配额。这条是 `--limit` 截断能用的前提 —— RACD 只跑
       0.28 个 epoch，若各组成块排列，截断处会整组整组地丢。
    3. **DDP 步数对齐**：run 按 `j % world_size` 分给各 rank，再把所有 rank 的长度统一
       补到最长（短的从自己头部循环补尾）并截到 `bs` 的整数倍。study 级分片长度不齐会
       让短 rank 先去 barrier、长 rank 还在 backward，600s 后 NCCL watchdog 打死整个
       job（2026-10-25 s4-2B 实测）—— 这里同款处理。
    """

    def __init__(self, ds, *, seed=0, bs=1, rank=0, world_size=1,
                 tau=0.5, min_frac=0.005, max_repeat=8, only_kind=None, tot=None):
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f'tau={tau} 应在 [0,1]（0=均匀按组数，1=按行数即全局均匀）')
        if max_repeat < 1:
            raise ValueError(f'max_repeat={max_repeat} < 1')
        if only_kind is not None and getattr(ds, 'kind_of', None) is None:
            raise ValueError('only_kind 需要 ds.kind_of —— 只有 MultiViewDataset 会建它')
        self.ds = ds
        self.seed, self.epoch = int(seed), 0
        self.bs, self.rank, self.world_size = int(bs), int(rank), int(world_size)
        self.tau, self.min_frac, self.max_repeat = float(tau), float(min_frac), int(max_repeat)
        self.only_kind = only_kind
        self.tot = None if tot is None else int(tot)
        self._skip = 0
        self.by_group = collections.OrderedDict()
        for i, (g, s) in enumerate(zip(ds.group_of, ds.study_of)):
            if only_kind is not None and ds.kind_of[i] != only_kind:
                continue
            self.by_group.setdefault(g, collections.OrderedDict()).setdefault(s, []).append(i)
        if not self.by_group:
            raise ValueError(f'only_kind={only_kind} 在这份数据里一行都没有 —— '
                             '题型名单与数据不同源？')
        self.names = sorted(self.by_group)
        self.sizes = {g: sum(len(v) for v in self.by_group[g].values()) for g in self.names}

    def quotas(self):
        """-> {组名: 本 epoch 该发多少题}。地板与封顶都在 `self` 的比例参数上。"""
        tot = self.tot if self.tot is not None else sum(self.sizes.values())
        w = {g: self.sizes[g] ** self.tau for g in self.names}
        sw = sum(w.values()) or 1.0
        floor = int(round(self.min_frac * tot))
        out = {}
        for g in self.names:
            q = max(floor, int(round(w[g] / sw * tot)))
            out[g] = min(q, self.sizes[g] * self.max_repeat)
        return out

    def _runs_of(self, g):
        return list(self.by_group[g].values())

    def _run_seq(self):
        """发牌序列：长度 = 各组 run 数之和，run 级交错（见类 docstring 约定 2）。"""
        gen = torch.Generator()
        gen.manual_seed(self.seed * 1000003 + self.epoch)
        q = self.quotas()
        tokens, cyc = [], {}
        for g in self.names:
            runs = self._runs_of(g)
            avg = max(1.0, self.sizes[g] / len(runs))
            n_tok = max(1, int(round(q[g] / avg)))
            tokens += [g] * n_tok
            cyc[g] = self._cyclic(runs, gen)
        perm = torch.randperm(len(tokens), generator=gen).tolist()
        return [next(cyc[tokens[p]]) for p in perm]

    @staticmethod
    def _cyclic(runs, gen):
        """每轮重新洗牌、无限发（配额超过组容量时就是有界复读，上限 = max_repeat）。"""
        while True:
            perm = torch.randperm(len(runs), generator=gen).tolist()
            for p in perm:
                yield runs[p]

    def _rank_lens(self, seq):
        lens = [0] * self.world_size
        for j, r in enumerate(seq):
            lens[j % self.world_size] += len(r)
        return lens

    def _target_len(self, seq):
        return (max(self._rank_lens(seq)) // self.bs) * self.bs

    def __len__(self):
        return self._target_len(self._run_seq())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.ds, 'set_view_epoch'):
            self.ds.set_view_epoch(self.epoch)

    def set_skip(self, n):
        """下一次迭代开始前跳过 n 条样本；只生效一次（续跑快进）。"""
        self._skip = max(0, int(n))

    def _indices(self, seq, target):
        mine = [i for j, r in enumerate(seq) if j % self.world_size == self.rank for i in r]
        n = 0
        while n < target:
            for i in mine:
                if n >= target:
                    return
                n += 1
                yield i

    def __iter__(self):
        seq = self._run_seq()
        it = self._indices(seq, self._target_len(seq))
        skip, self._skip = self._skip, 0
        if skip:
            it = itertools.islice(it, skip, None)
        yield from it


class KindBalancedSampler(Sampler):
    """题型平权：每个题型各跑一份 `GroupStratifiedSampler`，再把三条序列按比例交错。

    为什么要另起一层，而不是在 `GroupStratifiedSampler` 里加权：那层的配额是
    `q_g ∝ n_g^tau`，而题型在一个绑定器官组**内部**是混的 —— 在组内按题型重抽只能改
    **组内**比例，跨组的份额仍然跟着行数走，做不到平权。三条独立子序列各自取 `T_k`
    的配额才是精确的。

    配额分配 `T_k ∝ n_k^alpha`：
      alpha=1 ⇒ `T_k = n_k`，与不套这一层时同一个目标（本类只在 alpha<1 时才被构造）；
      alpha=0 ⇒ `T_k = 总行数/3`，三个题型拿到一样的曝光。

    动机是「训练权重 ≠ 报数权重」：报数按题型平权（noul mBAcc / choice mF1 /
    score MAE-QWK 各一列），训练行数却是 66.7% / 28.5% / 4.7%。
    🔴 代价写在明处：alpha=0 时 score 那 2,690 行每 epoch 要被过 ~7 遍（上限
    `max_repeat=8`）—— 那 2,690 道题会被背下来，测试集上可能表现为「训练 loss 掉、
    score 指标不动或变差」。所以 `alpha` 是连续旋钮，先跑 0.5 再看。

    交织：每一步从「已发比例最低」的题型取一条 ⇒ **任意前缀**的三类比例都贴着目标
    （与基类的 run 级交错同理，`--limit` 截断才不偏）。每条子序列自己已经按 `bs` 取整
    并按 rank 对齐过（`_target_len`），所以交错后各 rank 长度仍然相等 —— NCCL 不挂的
    前提在这里同样是靠继承来的，不是重新实现的。
    """

    def __init__(self, ds, *, kinds=('noul', 'choice', 'score'), alpha=0.0, **kw):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'alpha={alpha} 应在 [0,1]（1=按行数即现状，0=题型等分）')
        ko = getattr(ds, 'kind_of', None)
        if ko is None:
            raise ValueError('KindBalancedSampler 需要 ds.kind_of —— '
                             '只有 MultiViewDataset 会建它')
        cnt = collections.Counter(ko)
        present = [k for k in kinds if cnt.get(k)]
        if not present:
            raise ValueError(f'数据里一个 {kinds} 的题都没有 —— 名单与数据不同源？')
        n_all = int(sum(cnt.values()))
        w = {k: float(cnt[k]) ** alpha for k in present}
        sw = sum(w.values())
        self.ds = ds
        self.kinds = tuple(present)
        self.alpha = float(alpha)
        self.kind_rows = {k: int(cnt[k]) for k in present}
        self.targets = {k: max(1, int(round(n_all * w[k] / sw))) for k in present}
        self.sub = {k: GroupStratifiedSampler(ds, only_kind=k, tot=self.targets[k], **kw)
                    for k in present}
        self.sizes = {}
        for s in self.sub.values():
            for g, n in s.sizes.items():
                self.sizes[g] = self.sizes.get(g, 0) + n
        self._skip = 0
        self._cache = None

    def quotas(self):
        q = {}
        for s in self.sub.values():
            for g, n in s.quotas().items():
                q[g] = q.get(g, 0) + n
        return q

    def set_epoch(self, epoch):
        self._cache = None
        first = next(iter(self.sub.values()))
        first.set_epoch(epoch)
        for s in self.sub.values():
            if s is not first:
                s.epoch = int(epoch)

    def set_skip(self, n):
        """下一次迭代开始前跳过 n 条**交织后**的样本；只生效一次（续跑快进）。"""
        self._skip = max(0, int(n))

    def _mixed(self):
        if self._cache is None:
            lists = {}
            for k, s in self.sub.items():
                seq = s._run_seq()
                lists[k] = list(s._indices(seq, s._target_len(seq)))
            pos = dict.fromkeys(lists, 0)
            out = []
            while True:
                live = [k for k in lists if pos[k] < len(lists[k])]
                if not live:
                    break
                k = min(live, key=lambda j: pos[j] / len(lists[j]))
                out.append(lists[k][pos[k]])
                pos[k] += 1
            self._cache = out
        return self._cache

    def __len__(self):
        return len(self._mixed())

    def __iter__(self):
        it = iter(self._mixed())
        skip, self._skip = self._skip, 0
        if skip:
            it = itertools.islice(it, skip, None)
        yield from it
