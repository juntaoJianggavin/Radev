"""题型分组的决策损失：Intern-Decision 的硬标签项 + Vision-Jev 的质量/BCE/RPS。

逐题候选分数 `scores[B,K]`（无效候选为 -inf）在题型内合成：

* `choice`：`λ_ce·CE + λ_mass·(−log Σ_{k∈A} p_k) + λ_brier·Σ_k(p_k−1[k∈A])²`
  —— `A` 是可接受候选集合。单 gold 时质量项退化为 CE、Brier 退化为 one-hot 式；
  只有导出里带 `accept_mask`（多读者都判 S 的候选）时它才真正放宽“唯一正确项”。
* `noul`：`BCE(p_yes, y)`。它与“No/Yes 两个候选的 CE”在数学上等价，
  取 BCE 只是为了和 Vision-Jev 的读出口径对齐。列序恒为 0=No / 1=Yes；
  符号版的 a/b 同列序（b 是 Yes），所以这里不认字面键名。
* `score`：`NLL + λ_rps·RPS`。五档是有序的；RPS 只看累积质量，
  猜偏一档比猜偏三档罚得轻，且最后一位阈值恒等、贡献为 0。这里用未除以
  `K−1` 的和式（常数因子吸收进 λ_rps）；报数口径的 RPS 另在 metrics 里定。

温度校准不在这里：训练后按题型在独立 calibration split 上拟合 T，
并要求 argmax 不变（见任务 #84）。本模块只负责训练损失。

`racd_loss`（`--loss racd`）是 `combined_loss` 之上的**后置阶段**：SFT 用它，RACD
阶段 1 把同一份逐视图 proper score 当 task 项，再加两条视图之间的比较 hinge。
`combined_loss` 本身一字不改 —— SFT 路径逐位不变。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

CHOICE_CE = 0.5
CHOICE_MASS = 0.5
CHOICE_BRIER = 0.1
SCORE_RPS = 0.5


def _logp(scores: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(scores, dim=-1)


def choice_loss(scores, target, accept_mask=None, *, lam_ce=CHOICE_CE,
                lam_mass=CHOICE_MASS, lam_brier=CHOICE_BRIER):
    """choice：硬标签 CE + 可接受集合质量和 + Brier。"""
    logp = _logp(scores)
    p = logp.exp()
    onehot = F.one_hot(target, scores.size(-1)).to(p.dtype)
    ce = F.nll_loss(logp, target)

    if accept_mask is None:
        mask_b = onehot.to(torch.bool)
    else:
        mask_b = accept_mask.to(torch.bool)
        if mask_b.shape != scores.shape:
            raise ValueError(f'accept_mask {tuple(mask_b.shape)} != scores {tuple(scores.shape)}')
        if not bool(mask_b.any(dim=-1).all()):
            raise ValueError('accept_mask 有样本一个可接受候选都没有')
        if bool((~mask_b.gather(1, target[:, None])).any()):
            raise ValueError('accept_mask 没把 target 包进去 —— 唯一正确项必须可接受')
    log_z = torch.logsumexp(scores, dim=-1)
    log_mass = torch.logsumexp(scores.masked_fill(~mask_b, float('-inf')), dim=-1) - log_z
    mass = -log_mass.mean()
    brier = ((p - mask_b.to(p.dtype)) ** 2).sum(dim=-1).mean()
    return lam_ce * ce + lam_mass * mass + lam_brier * brier


def noul_loss(scores, target):
    """noul：对 Yes 的概率做 BCE（与两候选 CE 等价）。

    混批时 `scores` 会按 batch 内最大候选数补 `-inf` 到同一宽度，所以这里不能
    断言宽度恰为 2 —— 只要求前两列有效、其后全是被掩的填充。softmax 里这些
    `-inf` 不改变 No/Yes 的概率。
    """
    if scores.size(-1) < 2:
        raise ValueError(f'noul 的候选数至少为 2（No, Yes），实为 {scores.size(-1)}')
    if not bool(torch.isfinite(scores[:, :2]).all()):
        raise ValueError('noul 的前两列（No, Yes）里有 -inf —— 候选被长答案剔除逻辑误伤')
    if scores.size(-1) > 2 and bool(torch.isfinite(scores[:, 2:]).any()):
        raise ValueError('noul 行在第 3 列以后出现有效候选 —— 答案空间接错了')
    p = _logp(scores).exp()
    with torch.autocast('cuda', enabled=False):
        p32 = p.float()
        return F.binary_cross_entropy(p32[:, 1].clamp(1e-6, 1 - 1e-6),
                                      target.to(p32.dtype))


def score_loss(scores, target, *, lam_nll=1.0, lam_rps=SCORE_RPS):
    """score：有序档位的 NLL + RPS。"""
    logp = _logp(scores)
    p = logp.exp()
    nll = F.nll_loss(logp, target)
    cdf = p.cumsum(dim=-1)
    target_cdf = F.one_hot(target, scores.size(-1)).to(p.dtype).cumsum(dim=-1)
    rps = ((cdf - target_cdf) ** 2).sum(dim=-1).mean()
    return lam_nll * nll + lam_rps * rps


def combined_loss(scores, target, kinds, accept_mask=None, row_keep=None):
    """逐样本按题型取损失，再对 batch 取平均。

    `kinds[i] ∈ {'choice','noul','score'}`；`accept_mask` 仅对 choice 生效。
    逐样本计算而不是先按题型分组，避免 batch 内题型配比把某一族的权重放大。

    `row_keep` = `[B]` bool：哪些行进答案监督。给 `racd_loss` 用来把反事实视图摘出去
    （同一行上 CE 要 `c→1`、rank 要 `c ≤ c_full − m`，两个要求符号相反，见 `task_views`）。
    不给 = 全行同权（历史口径，已跑的两条 RACD 臂是这一种）。
    """
    if scores.ndim != 2 or target.ndim != 1:
        raise ValueError(f'scores {tuple(scores.shape)} / target {tuple(target.shape)} '
                         '应为 [B,K] / [B]')
    if len(kinds) != scores.size(0):
        raise ValueError(f'kinds 有 {len(kinds)} 项，scores 有 {scores.size(0)} 行')
    if accept_mask is not None and accept_mask.shape != scores.shape:
        raise ValueError(f'accept_mask {tuple(accept_mask.shape)} != scores {tuple(scores.shape)}')
    if row_keep is not None and row_keep.shape != (scores.size(0),):
        raise ValueError(f'row_keep {tuple(row_keep.shape)} != [B]')

    losses = []
    for i, kind in enumerate(kinds):
        if row_keep is not None and not bool(row_keep[i]):
            continue
        s = scores[i:i + 1]
        y = target[i:i + 1]
        m = None if accept_mask is None else accept_mask[i:i + 1]
        if kind == 'choice':
            losses.append(choice_loss(s, y, m))
        elif kind == 'noul':
            losses.append(noul_loss(s, y))
        elif kind == 'score':
            losses.append(score_loss(s, y))
        else:
            raise ValueError(f'未知题型 {kind!r}（应为 choice/noul/score）')
    if not losses:
        raise ValueError('combined_loss 收到空 batch（或 row_keep 把所有行都摘掉了）')
    return torch.stack(losses).mean()


RACD_RANK = 0.2
RACD_NULL = 0.1
RACD_M_RANK = 0.05
RACD_M_NULL = 0.03
RACD_BLIND = 0.0
RACD_M_BLIND = 0.20

RACD_CAL = 0.0
RACD_CAL_MODE = 'flat'
RACD_CAL_PI_LO = 0.55
RACD_CAL_PI_HI = 0.85
RACD_CAL_PI_W = 0.2


def _bgap_key(t):
    return 'bgap_lt_' + f'{t:g}'.replace('.', 'p')


RACD_BGAP_TH = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60)
RACD_BGAP_KEYS = tuple(_bgap_key(t) for t in RACD_BGAP_TH)
RACD_TASK_VIEWS = 'all'

RACD_DIAG = ('task', 'rank', 'null', 'c_full', 'c_tgt', 'c_dist', 'q_full', 'q_tgt',
             'd_tgt', 'd_tgt_gated', 'd_dist', 'gate_frac', 'viol_rank', 'viol_null',
             'task_keep')

RACD_DIAG_W = {'task': None, 'c_full': None, 'n_tgt': None, 'n_dist': None,
               'rank': 'n_tgt', 'c_tgt': 'n_tgt', 'q_full': 'n_tgt', 'q_tgt': 'n_tgt',
               'd_tgt': 'n_tgt', 'd_tgt_gated': 'n_tgt', 'gate_frac': 'n_tgt',
               'viol_rank': 'n_tgt',
               'null': 'n_dist', 'c_dist': 'n_dist', 'd_dist': 'n_dist',
               'viol_null': 'n_dist',
               'n_blind': None, 'n_blind_gate': None,
               'c_blind': 'n_blind', 'q_blind': 'n_blind', 'd_blind': 'n_blind',
               'gate_frac_blind': 'n_blind',
               'blind': 'n_blind_gate', 'd_blind_gated': 'n_blind_gate',
               'viol_blind': 'n_blind_gate',
               **{k: 'n_blind_gate' for k in RACD_BGAP_KEYS},
               'task_keep': None,
               'cal': None, 'cal_y': None, 'cal_w': None, 'cal_off': None}


def _conf(scores: torch.Tensor) -> torch.Tensor:
    """`[N,K]` -> `[N]` 最大候选概率。`-inf` 填充列经 softmax 恒 0，不影响 max。"""
    return _logp(scores).max(dim=-1).values.exp()


def _quality(scores: torch.Tensor, target: torch.Tensor, kinds) -> torch.Tensor:
    """`[N,K]` -> `[N]` 决策质量 ∈ [0,1]（越大越好）。

    有效宽度按行取 `isfinite` 计数：混批时 noul 行被补 `-inf` 到 batch 最大候选数，
    score 行的 `K−1` 必须是 4（五档）而不是补出来的宽度，否则「差一档」的罚会被稀释。
    """
    p = _logp(scores).exp()
    pred = p.argmax(dim=-1)
    nvalid = torch.isfinite(scores).sum(dim=-1)
    denom = (nvalid - 1).clamp(min=1).to(torch.float32)
    hit = (pred == target).to(torch.float32)
    close = 1.0 - (pred - target).abs().to(torch.float32) / denom
    is_score = torch.tensor([k == 'score' for k in kinds], device=scores.device)
    return torch.where(is_score, close, hit)


def racd_loss(scores, target, kinds, view_kind, *, n_views,
              lam_rank=RACD_RANK, lam_null=RACD_NULL,
              m_rank=RACD_M_RANK, m_null=RACD_M_NULL, accept_mask=None,
              task_views=RACD_TASK_VIEWS,
              lam_blind=RACD_BLIND, m_blind=RACD_M_BLIND,
              lam_cal=RACD_CAL, cal_mode=RACD_CAL_MODE, cal_prior=None,
              cal_pi_lo=RACD_CAL_PI_LO, cal_pi_hi=RACD_CAL_PI_HI,
              cal_pi_w=RACD_CAL_PI_W):
    """-> `(loss, diag)`；`diag` 是 `RACD_DIAG` 键序的 float dict（可直接进 jsonl）。

    `scores [B·M, K]`、`view_kind` 同长；行序 = 组主序（`MultiViewDataset.collate` 的
    约定：行 `r` 属组 `r // M`、视图槽 `r % M`），槽 0 恒 `full`。抽不出反事实的槽会
    退回 `full`，那种组按 kind 识别后**跳过**比较项（不是当作零损失混进均值）。

    `task_views` 决定**答案监督给到哪些视图**（三值）：

    * `'all'` —— 历史口径：每个视图的读出都要是校准的，屏蔽视图与整卷视图同权。已跑
      的两条臂（`out_racd/2B-oev-*`）是这一种。**它的已知毛病**：`target_mask` 行上
      CE 要求 `c→1`，而 rank 项要求 `c_tgt ≤ c_full − m_rank`，两个要求作用在同一份
      logits 上且符号相反 —— 门控为真的组里它们在互相抵消，权重比约 1 : 0.0005。
    * `'no_target'` —— `target_mask` 行退出答案监督，只由 rank 项塑形。这是 RAC
      （arXiv:2605.16999）/RLCD（arXiv:2609.38850）的原式：corrupted 视图进比较项，
      不进答案损失。
    * `'non_gated'` —— 折中：遮蔽后**答案没变坏**的组（`q_full == q_tgt`，门控为假）
      继续吃 CE（这条监督与不变性不冲突，就是增广），**答案变坏了**的组（门控为真）
      退出 CE，把置信度让给 rank 项。矛盾被逐组消除，而不是整条视图一刀切。

    `distractor` 视图在所有口径下都吃 CE：屏蔽无关器官本来就该答得一样（`null` 项要
    的就是不变），两项在这条视图上同向，没有矛盾。

    ── 盲视图项（`lam_blind>0`；默认 0 ⇒ 整段不生效，既有臂逐位不变）──────────────

    `blind` 视图（`views.VIEW_BLIND`，恒占最后一槽）= **全部**在场视觉槽一次屏掉，
    即「只看题面、不看图」的那一端。它与 rank/null 两项的区别是语义上的：

    * rank 要 `c_full − c_tgt ≥ m_rank`（拿掉**绑定**证据，信心该掉）；
      null 要 `c_full − c_dist ≤ m_null`（拿掉**无关**证据，信心不该掉）；
      blind 要 `c_full − c_blind ≥ m_blind`（拿掉**全部**证据，信心该掉到只剩先验）。
    * blind 是这三条里唯一一条**抓语言先验**的：本项目「不看影像按每题经验众数就有
      macro acc 0.7605」，那批题在盲视图下信心照样高 ⇒ `c_full − c_blind ≈ 0` ⇒ 违约 ⇒
      这份信心不是证据支撑的。rank 项抓的是「绑定器官的证据有没有被用上」，抓不到
      「答案从先验里来」这件事（先验强的题往往绑定器官也确实有用，门控为假）。
    * 门控同 rank 的写法：`gate_b = (q_full > q_blind)`，只在「盲掉真的把答案打坏了」的组上
      要求信心差。不加闸就是在「盲了也答对」的题上硬造置信度差 —— 那种题本来就该有信心。

    🔴 **`c_full` 在这一项里 detach**。报数口径读的是 full 视图的信心，而 hinge 不 detach 的话
    梯度会同时把 `c_full` 往上推 —— 那正是 13 条 GRPO 臂赔掉校准的方向（`topECE ≈ conf − acc`，
    信心升、准确率不动 ⇒ ECE 直接变差）。detach 之后这一项对 full 视图的信心**零直接梯度**，
    只剩「盲视图必须更不自信」这一头；权重照样更新（`c_blind` 与 `c_full` 共享全部参数），
    改的是「信心必须依赖证据在场」这件事，而不是「信心必须高」。
    🔴 rank/null 两项**不**跟着改：它们的既有读数已落盘（armA/B/B1/B2/B3/C 六条），
    动 detach 就动了那六条臂的数值，不再是逐位复现。

    🔴 **盲视图无条件退出答案监督**（不看 `task_views`）：要求「什么证据都没有」的读出答对，
    就是在教语言先验 —— 与这一项存在的理由正好相反。它和 target/distractor 不同：那两个是
    「证据被削了一部分」，答对仍是合法要求（所以 `'all'` 口径下它们吃 CE）；blind 是「证据
    全没了」，答对只能靠猜。⇒ `task_keep` 在 `VIEWS=4 + blind` 下 ≈ 3/4，而**吃 CE 的那三行
    正是 `VIEWS=3` 那三个视图** ⇒ 与 armC/armD 的答案监督行集逐位同构。

    ── 逐行可靠性项（`lam_cal>0`；默认 0 ⇒ 整段不生效，既有六条臂逐位不变）────────

    `lam_cal · mean_rows v(π_q)·(p_max − y)²`，`y = 1[argmax == gold]`。训练期 gold 已知，
    这是 p_max → E[y|x] 的 strictly proper 目标：**直接拿 eval 口径当损失**，不是事后 TS，
    也不碰 argmax 之外的排序（argmax 是下标比较，天生无梯度，梯度只经 p_max 那一路）。
    `y` 三型一律是**精确命中**，score 不用 `_quality` 的有序档距：判据里的 `偏差_score` /
    `信心ECE_score` 就是按 `acc_score`（精确命中）算的，损失与报数口径必须同一个 0/1。

    * 为什么必须有它：`out_logs/ts_ceiling.py` 实测 armD 上做完美温度标定只能再吃掉
      noul 8% / choice 12% / score 31% 的 ECE，**残差占比 69~92%** ⇒ 残余在形状不在尺度，
      任何单参数（或仿射）读出都拿不到，只有把「信心 = 答对概率」写进损失才动得到形状。
    * 🔴 **只在槽 0（full 视图）上算**：反事实/盲视图的读出天然是「证据被削掉之后」的分布，
      把它们的 p_max 也拉向 y 会与 rank/null/blind 三项符号相反地互相抵消（`task_views='all'`
      下 CE 与 rank 已经在这个坑里，见上面那段 1 : 0.0005 的权重比）。报数读的是 full 视图，
      要校准的也就是 full 视图。
    * 与 CE 的对抗是本项唯一真风险：CE 要 p_max→1，这一项要 p_max→E[y]。处理方式不在这里，
      在调用方 —— `--racd-cal-warm` 做两阶段点火（前 N 步 λ_cal=0 走纯 armF，之后线性升到
      目标 λ），所以传进来的 `lam_cal` 是**已经乘好 ramp 的当步值**。
    * `cal_mode='pi'` 用训练侧先验 π_q（`--racd-prior-table`，只由 train+val 标签算、不碰 test
      ⇒ 不构成 oracle-on-test）给行加权：带内 `v=1`，带外 `v=cal_pi_w`（**不是 0**：摘掉带外
      行等于同时改了 batch 构成，与 armF 就不止差一项了）。`cal_mode='flat'` 是它的对照。
    * 诊断键 `cal/cal_y/cal_w/cal_off` 条件挂出（λ_cal=0 时 train_log.jsonl 逐位不变，与 blind
      那六个键同一条纪律）。`cal_y` = full 视图行的答对率，`cal_off` = π 带外行占比。
      这两列是「质量 >3% 的单元不得靠整体压信心换 ECE」那条护栏的训练侧对应物：
      `cal_y` 明显掉而 `cal` 不降 ⇒ 这一项在做整体制平，不是在校准，该降 λ 而不是加步数。
    """
    from .views import VIEW_BLIND, VIEW_DISTRACTOR, VIEW_FULL, VIEW_TARGET
    if task_views not in ('all', 'no_target', 'non_gated'):
        raise ValueError(f'task_views={task_views!r} 应为 all/no_target/non_gated')
    if lam_blind < 0:
        raise ValueError(f'lam_blind={lam_blind} < 0 —— 翻号就变成「盲了还信心满满」受奖')
    if m_blind < 0:
        raise ValueError(f'm_blind={m_blind} < 0 —— 间隔为负等于要求盲视图比整卷更自信')
    if lam_cal < 0:
        raise ValueError(f'lam_cal={lam_cal} < 0 —— 翻号就变成「信心离答对概率越远越好」')
    if lam_cal > 0 and cal_mode not in ('flat', 'pi'):
        raise ValueError(f'cal_mode={cal_mode!r} 应为 flat/pi')
    if cal_mode == 'pi' and lam_cal > 0 and cal_prior is None:
        raise ValueError('cal_mode=pi 要 cal_prior（逐组 π_q 向量）—— '
                         '没有它 v(π_q) 无从算起，静默退回 flat 会把两条臂混成一个口径')
    if cal_pi_hi < cal_pi_lo:
        raise ValueError(f'cal_pi_hi={cal_pi_hi} < cal_pi_lo={cal_pi_lo} —— 带是空集，'
                         '等于只在带外加权（方向反了）')
    M = int(n_views)
    if M < 2:
        raise ValueError(f'n_views={M} < 2 —— 单视图没有可比的两端，用 --loss combined')
    if scores.ndim != 2 or scores.size(0) % M:
        raise ValueError(f'scores {tuple(scores.shape)} 的行数不是 n_views={M} 的整数倍')
    if len(view_kind) != scores.size(0):
        raise ValueError(f'view_kind 有 {len(view_kind)} 项，scores 有 {scores.size(0)} 行')
    B = scores.size(0) // M
    if any(view_kind[g * M] != VIEW_FULL for g in range(B)):
        raise ValueError('槽 0 必须是 full 视图 —— 两项比较损失都以它为基准')
    for g in range(B):
        ks = kinds[g * M:(g + 1) * M]
        if len(set(ks)) != 1:
            raise ValueError(f'组 {g} 的 {M} 个视图题型不一致 {ks} —— 组主序被破坏了')
    if target.view(B, M).ne(target.view(B, M)[:, :1]).any():
        raise ValueError('同一组的 M 个视图 target 不一致 —— 视图只改证据，不改答案')

    s = scores.float()
    c = _conf(s).view(B, M)
    q = _quality(s, target, kinds).view(B, M)

    t_cpu = [-1] * B
    d_cpu = [-1] * B
    b_cpu = [-1] * B
    for g in range(B):
        for m in range(M):
            k = view_kind[g * M + m]
            if k == VIEW_TARGET and t_cpu[g] < 0:
                t_cpu[g] = m
            elif k == VIEW_DISTRACTOR and d_cpu[g] < 0:
                d_cpu[g] = m
            elif k == VIEW_BLIND and b_cpu[g] < 0:
                b_cpu[g] = m
    t_slot = torch.tensor(t_cpu, dtype=torch.long, device=s.device)
    d_slot = torch.tensor(d_cpu, dtype=torch.long, device=s.device)
    b_slot = torch.tensor(b_cpu, dtype=torch.long, device=s.device)

    z = s.new_zeros(())
    l_rank, l_null = z, z
    d_tgt = d_tgt_g = d_dist = gate_frac = viol_rank = viol_null = z
    c_tgt = c_dist = q_full = q_tgt = z
    n_tgt = n_dist = 0
    l_blind = z
    l_cal = z
    cal_y = cal_w = cal_off = z
    c_blind = q_blind = d_blind = d_blind_g = gate_frac_blind = viol_blind = z
    bgap = None
    n_blind = n_blind_gate = 0
    cvals = None

    rows_t = (t_slot >= 0).nonzero(as_tuple=True)[0]
    if rows_t.numel():
        n_tgt = int(rows_t.numel())
        ix = t_slot[rows_t].unsqueeze(1)
        c_f, q_f = c[rows_t, 0], q[rows_t, 0]
        c_t = c[rows_t].gather(1, ix).squeeze(1)
        q_t = q[rows_t].gather(1, ix).squeeze(1)
        gate = (q_f > q_t).to(torch.float32)
        viol = F.relu(m_rank - (c_f - c_t))
        l_rank = (viol * gate).sum() / gate.sum().clamp(min=1.0)
        d = c_f - c_t
        d_tgt = d.mean()
        gate_frac = gate.mean()
        viol_rank = ((viol > 0).to(torch.float32) * gate).sum() / gate.sum().clamp(min=1.0)
        c_tgt, q_full, q_tgt = c_t.mean(), q_f.mean(), q_t.mean()
        d_tgt_g = (d * gate).sum() / gate.sum().clamp(min=1.0)
    rows_d = (d_slot >= 0).nonzero(as_tuple=True)[0]
    if rows_d.numel():
        n_dist = int(rows_d.numel())
        c_f = c[rows_d, 0]
        c_d = c[rows_d].gather(1, d_slot[rows_d].unsqueeze(1)).squeeze(1)
        d_dist = (c_f - c_d).mean()
        l_null = F.relu((c_f - c_d) - m_null).mean()
        viol_null = ((c_f - c_d) > m_null).to(torch.float32).mean()
        c_dist = c_d.mean()

    rows_b = (b_slot >= 0).nonzero(as_tuple=True)[0]
    if rows_b.numel():
        n_blind = int(rows_b.numel())
        ixb = b_slot[rows_b].unsqueeze(1)
        c_fb, q_fb = c[rows_b, 0], q[rows_b, 0]
        c_b = c[rows_b].gather(1, ixb).squeeze(1)
        q_b = q[rows_b].gather(1, ixb).squeeze(1)
        gate_b = (q_fb > q_b).to(torch.float32)
        nb = gate_b.sum().clamp(min=1.0)
        d_b = c_fb - c_b
        ell_b = F.relu(m_blind - (c_fb.detach() - c_b))
        l_blind = (ell_b * gate_b).sum() / nb
        viol_blind = ((ell_b > 0).to(torch.float32) * gate_b).sum() / nb
        bgap = [(((d_b < t).to(torch.float32)) * gate_b).sum() / nb for t in RACD_BGAP_TH]
        n_blind_gate = int(gate_b.sum())
        gate_frac_blind = gate_b.mean()
        d_blind = d_b.mean()
        d_blind_g = (d_b * gate_b).sum() / nb
        c_blind, q_blind = c_b.mean(), q_b.mean()

    if lam_cal > 0:
        rows_f = torch.arange(B, device=s.device) * M
        lp_f = _logp(s[rows_f])
        c_f0 = lp_f.max(dim=-1).values.exp()
        yv = (lp_f.argmax(dim=-1) == target[rows_f]).to(torch.float32)
        if cal_mode == 'flat':
            w = torch.ones(B, device=s.device, dtype=s.dtype)
            cal_off = z
        else:
            pr = cal_prior.reshape(-1).to(device=s.device, dtype=s.dtype)
            if pr.numel() != B:
                raise ValueError(f'cal_prior 长 {pr.numel()} != 组数 {B} —— 逐组 π_q 与批次对不上')
            inb = (pr >= cal_pi_lo) & (pr <= cal_pi_hi)
            w = torch.where(inb, torch.ones_like(pr), torch.full_like(pr, float(cal_pi_w)))
            cal_off = (~inb).to(torch.float32).mean()
        cal = (w * (c_f0 - yv) ** 2).sum() / w.sum().clamp(min=1.0)
        l_cal = lam_cal * cal
        cal_y = yv.mean()
        cal_w = w.mean()
        cvals = torch.stack([cal.detach(), cal_y.detach(), cal_w.detach(), cal_off.detach()])

    keep = torch.ones(B * M, dtype=torch.bool, device=s.device)
    if task_views != 'all' and rows_t.numel():
        gix = rows_t * M + t_slot[rows_t]
        if task_views == 'no_target':
            keep[gix] = False
        else:
            keep[gix[gate > 0]] = False
    if rows_b.numel():
        keep[rows_b * M + b_slot[rows_b]] = False
    task = combined_loss(scores, target, kinds, accept_mask, row_keep=keep)

    loss = task + lam_rank * l_rank + lam_null * l_null + lam_blind * l_blind + l_cal
    c_full = c[:, 0].mean()
    vals = torch.stack([task.detach(), l_rank.detach(), l_null.detach(), c_full.detach(),
                        c_tgt.detach(), c_dist.detach(), q_full.detach(), q_tgt.detach(),
                        d_tgt.detach(), d_tgt_g.detach(), d_dist.detach(),
                        gate_frac.detach(), viol_rank.detach(), viol_null.detach(),
                        keep.to(torch.float32).mean().detach()])
    diag = {k: round(v, 6) for k, v in zip(RACD_DIAG, vals.tolist())}
    diag['n_tgt'], diag['n_dist'] = n_tgt, n_dist
    if bgap is not None:
        bvals = torch.stack([l_blind.detach(), c_blind.detach(), q_blind.detach(),
                             d_blind.detach(), d_blind_g.detach(),
                             gate_frac_blind.detach(), viol_blind.detach()]
                            + [x.detach() for x in bgap])
        bkeys = (('blind', 'c_blind', 'q_blind', 'd_blind', 'd_blind_gated',
                  'gate_frac_blind', 'viol_blind') + RACD_BGAP_KEYS)
        diag.update({k: round(v, 6) for k, v in zip(bkeys, bvals.tolist())})
        diag['n_blind'], diag['n_blind_gate'] = n_blind, n_blind_gate
    if cvals is not None:
        ckeys = ('cal', 'cal_y', 'cal_w', 'cal_off')
        diag.update({k: round(v, 6) for k, v in zip(ckeys, cvals.tolist())})
    return loss, diag


GRPO_SAMPLES = 8
GRPO_M_RANK = 0.05
GRPO_RANK = 0.2
GRPO_CORR = 0.3
GRPO_M_CORR = 0.05
GRPO_ALPHA = 0.0
GRPO_BETA = 0.04
GRPO_LAM_TASK = 0.0
GRPO_W_TASK = 1.0
GRPO_ADV_EPS = 1e-3

GRPO_KINDS = ('noul', 'choice', 'score')
_GRPO_KIND_ROW = tuple(f'{k}_{s}' for k in GRPO_KINDS
                       for s in ('share', 'r_task', 'adv_nz', 'hit', 'pi_max'))
_GRPO_KIND_CORR = tuple(f'{k}_{s}' for k in GRPO_KINDS
                        for s in ('viol_corr', 'gate_frac'))

GRPO_DIAG_BASE = ('task', 'pg', 'kl', 'r_task', 'r_rank', 'r_corr', 'r_mean',
                  'r_std', 'adv_nz', 'pi_max', 'hit', 'viol_rank', 'viol_corr',
                  'gate_corr')
GRPO_DIAG = GRPO_DIAG_BASE + _GRPO_KIND_ROW + _GRPO_KIND_CORR

GRPO_DIAG_COUNT = (('n_corr',) + tuple(f'n_corr_{k}' for k in GRPO_KINDS)
                   + tuple(f'n_row_{k}' for k in GRPO_KINDS))

GRPO_DIAG_W = dict.fromkeys(GRPO_DIAG)
GRPO_DIAG_W.update({'viol_corr': 'n_corr', 'gate_corr': 'n_corr'})
GRPO_DIAG_W.update({f'{k}_{s}': f'n_corr_{k}' for k in GRPO_KINDS
                    for s in ('viol_corr', 'gate_frac')})
GRPO_DIAG_W.update({f'{k}_{s}': f'n_row_{k}' for k in GRPO_KINDS
                    for s in ('r_task', 'adv_nz', 'hit', 'pi_max')})


def _gap_key(t):
    return 'gap_lt_' + f'{t:g}'.replace('.', 'p')


GRPO_GAP_TH = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
GRPO_GAP_KEYS = tuple(_gap_key(t) for t in GRPO_GAP_TH)
GRPO_DIAG_W.update({k: 'n_gate' for k in GRPO_GAP_KEYS})
GRPO_DIAG_W['gap_mean'] = 'n_gate'


def _rgap_key(t):
    return 'rgap_lt_' + f'{t:g}'.replace('.', 'p')


GRPO_RGAP_TH = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80)
GRPO_RGAP_KEYS = tuple(_rgap_key(t) for t in GRPO_RGAP_TH)
GRPO_DIAG_W.update({k: 'n_pair' for k in GRPO_RGAP_KEYS})
GRPO_DIAG_W['rgap_mean'] = 'n_pair'


EV_SAMPLES = 8
EV_LAM_COST = 0.02
EV_BETA_CAL = 0.1
EV_BETA_KL = 0.01
EV_LAM_TASK = 1.0
EVGRPO_DIAG_BASE = ('task', 'pg', 'kl', 'acq_cost', 'acq_brier', 'acq_correct',
                    'adv_nz', 'acq_bound_frac', 'acq_whole_frac', 'hit', 'pi_max',
                    'r_mean', 'r_std')
_EVGRPO_KIND_ROW = tuple(f'{k}_{s}' for k in GRPO_KINDS
                         for s in ('share', 'hit', 'adv_nz', 'pi_max'))
EVGRPO_DIAG = EVGRPO_DIAG_BASE + _EVGRPO_KIND_ROW
EVGRPO_DIAG_COUNT = tuple(f'n_row_{k}' for k in GRPO_KINDS)
EVGRPO_DIAG_W = dict.fromkeys(EVGRPO_DIAG)
EVGRPO_DIAG_W.update({f'{k}_{s}': f'n_row_{k}' for k in GRPO_KINDS
                      for s in ('hit', 'adv_nz', 'pi_max')})


def _reward(scores, act, target, kinds, accept_mask=None):
    """`act [N,G]`（采到的候选列号）-> `[N,G]` 奖励 ∈ [0,1]，越大越好。RAC 的 `r^task`。

    与 `_quality` 同一个口径，只是那里喂 argmax、这里喂采样：
    choice = 1[a∈A]（A 就是 `choice_loss` 的 accept_mask，多读者都判 S 的候选都可接受，
    没有 mask 时退化成 1[a=y]）；noul = 1[a=y]；score = 1−|a−y|/(K−1)（有序档位，
    差一档罚得轻）。K−1 按行取 `isfinite` 计数 —— 混批时 noul 行被补 `-inf` 到 batch
    最大宽度，score 行的分母必须是 4（五档）而不是补出来的宽度。
    """
    t = target.unsqueeze(1)
    hit = (act == t).to(torch.float32)
    nvalid = torch.isfinite(scores).sum(dim=-1)
    denom = (nvalid - 1).clamp(min=1).to(torch.float32).unsqueeze(1)
    close = 1.0 - (act - t).abs().to(torch.float32) / denom
    dev = scores.device
    is_score = torch.tensor([k == 'score' for k in kinds], device=dev).unsqueeze(1)
    r = torch.where(is_score, close, hit)
    if accept_mask is not None:
        is_choice = torch.tensor([k == 'choice' for k in kinds], device=dev).unsqueeze(1)
        r = torch.where(is_choice, accept_mask.to(torch.float32).gather(1, act), r)
    return r


def _rank_shaping(r_task, conf, m_rank):
    """RAC Eq.4–6 -> `(r^rank [N,G] ≤ 0, 违约对占比)`。全程 detach 的**奖励**，不是损失。

    `ℓ_ij = relu(m_rank − (c_i − c_j))` 只在 `r_i > r_j` 的对上算（原文的 `Q(x)`，
    `f_i=f_j=1` 对我们恒成立）。按原文 line 142 的口径，每个 rollout 累加**它参与的所有
    有效对**的违约：作为较优一方要够自信（`ℓ_ij`），作为较差一方不能太自信（`ℓ_ji`）。
    两个方向都是罚 ⇒ 返回负值。

    进来先 detach：RAC 的整形项是**奖励**（原文「only changes the reward signal」），
    不许自己开一条绕过组内归一化的梯度通路。`grpo_loss` 里它本来就跑在 `no_grad` 下，
    这里是让这条性质不依赖调用点。

    第三个返回值是 `d = c_i − c_j` 的 CDF 读数栈（见 GRPO_RGAP_TH）：
    `[n_pair, F(t) for t in GRPO_RGAP_TH, d 的均值]`，全程只读、不进梯度、
    **不碰 better/worse/ell/frac 任何一条数值**。`n_pair=0` 时栈里全是被 clamp 兜住的 0，
    由调用点看 `n_pair` 决定**整组键缺席**（不是写 0），与 corr 轴同一条纪律。
    """
    r_task, conf = r_task.detach(), conf.detach()
    gt = r_task.unsqueeze(2) > r_task.unsqueeze(1)
    d = conf.unsqueeze(2) - conf.unsqueeze(1)
    ell = F.relu(m_rank - d)
    better = (gt * ell).sum(dim=2)
    worse = (gt.transpose(1, 2) * ell.transpose(1, 2)).sum(dim=2)
    n_pair = gt.sum()
    viol = (gt & (ell > 0)).sum()
    npf = n_pair.clamp(min=1).to(torch.float32)
    frac = viol.to(torch.float32) / npf
    rgap = torch.stack(
        [n_pair.to(torch.float32)]
        + [((d < t) & gt).sum().to(torch.float32) / npf for t in GRPO_RGAP_TH]
        + [(d * gt).sum() / npf])
    return -(better + worse), frac, rgap


def _slot_of(view_kind, B, M, want):
    """每组第一个 `want` 视图的槽号，没有则 -1（纯 CPU 字符串逻辑，不上 GPU）。"""
    out = [-1] * B
    for g in range(B):
        for m in range(M):
            if view_kind[g * M + m] == want:
                out[g] = m
                break
    return out


def grpo_loss(scores, target, kinds, accept_mask=None, *, ref_scores=None,
              view_kind=None, n_views=0,
              n_samples=GRPO_SAMPLES, beta=GRPO_BETA, lam_task=GRPO_LAM_TASK,
              w_task=GRPO_W_TASK,
              lam_rank=GRPO_RANK, m_rank=GRPO_M_RANK,
              lam_corr=GRPO_CORR, m_corr=GRPO_M_CORR, alpha_sev=GRPO_ALPHA,
              gate_corr=True, sev=1.0, corr_cand='cross', task_views=None,
              view_weight='uniform'):
    """-> `(loss, diag)`；`diag` 是 `GRPO_DIAG` 键序的 float dict（可直接进 jsonl）。

    单视图：`scores [B,K]`（无效候选 `-inf`）、`target [B]`、`kinds` 长 B，口径与
    `combined_loss` 完全一致 —— 此时只有 r^task 与 r^rank 两条整形，`lam_corr` 必须为 0。
    多视图：给 `view_kind` + `n_views`，行序 = 组主序（`MultiViewDataset.collate` 的约定：
    行 `r` 属组 `r//M`、视图槽 `r%M`，槽 0 恒 `full`），与 `racd_loss` 同一套约定，
    r^corr 才有干净/退化两端可配。`ref_scores` 是参考策略在**同一批**上的逐候选分数
    （`beta=0` 时不需要）。

    `corr_cand` 定 r^corr 那个 hinge 比的是**谁的**概率：
    * `cross`（默认 = 已跑过的六条 RL 臂口径，逐位可复现）：`c0=π_full(full 视图自己采到的
      候选)`、`c1=π_mask(mask 视图自己采到的候选)` —— 两个**不同答案**的概率相减。
    * `same`：候选号取干净视图那一次抽样，两端都在它上取 π —— 与 `racd_loss` 的
      `c_f/c_t`（losses 里同一段 `gather(1, ix)`，同候选跨视图）同一个对法。
    选 `cross` 时 hinge 有个**结构性死角**：gate 要求「干净端答对、退化端答错」，
    而 `c1 − c0 + m > 0` 要求「退化端给那个错答案的信心 ≥ 干净端给对答案的信心 − m」；
    后者随模型变好而越来越难满足（干净端 π(gold)→1 时恒不成立），所以 r^corr 会在
    训练中途自己熄火 —— 实测在跑的臂 r_corr=−0.0099，只占 r_std=0.1605 的 6%。

    `task_views`（默认 `None` = 历史口径，逐位不变）决定**答案监督给到哪些视图**，
    取值与 `racd_loss` 同名参数一致（`all`/`no_target`/`non_gated`）。它存在的全部理由
    是 `corr_cand='same'` 下的一个**同 logit 冲突**：gate 为真意味着干净端采到的就是
    gold（`r0=1`），于是 hinge 压的是 `π_mask(gold)`，而 `lam_task` 的 CE 在同一个 logit
    上要求 `π_mask(gold)→1` —— 两项符号相反，corr 信号被部分抵消，事后分不清「机制无效」
    还是「机制被抵消」。`no_target` 是 RAC/RLCD 的原式（反事实视图只进比较项）；
    `non_gated` 只摘**组内任一次 rollout 开了闸**的那些行 —— 判据是 `gate.any(dim=1)`
    而不是 racd 的逐组标量，因为这里的 gate 逐 (组, 样本) 算，只要有一次开火冲突就已经
    落在那一份 logits 上了。`None` 与 `'all'` 等价，且 `row_keep` 保持 `None`（不是全 True
    的张量），所以既有的六条臂走的还是同一个调用。

    `w_task`（默认 1.0 = RAC Eq.11 原文，既有臂逐位不变）是 r̃ 里 r^task 的权重：
    `r̃ = w_task·r^task + λ_rank·r^rank + λ_corr·r^corr`。它存在的理由见 `GRPO_W_TASK`
    的注记 —— `_reward` 是答案型 ⇒ `E_p[r]` 对 p 线性 ⇒ 单纯形上顶点最优 ⇒ 这一项
    是**纯锐化**，与提校准反向，而 λ_task/β/λ_corr 都改不了它的方向，只能改走多快。
    `w_task=0` 时 pg 只携带 r^corr 的反事实信号；`_reward` **照常算**（gate 的
    `r0>r1` 与 `task_views='non_gated'` 的筛选都读 rt），diag 的 `r_task` 也照记原值
    ⇒ 跨臂可比，而 `r_mean` 从此等于 `w_task·r_task + λ_rank·r_rank + λ_corr·r_corr`。
    读数上会变的只有一处：整组一次都没开闸时 r̃≡0 ⇒ sd=0 ⇒ adv=0，所以 `adv_nz`
    会显著低于 w_task=1 的臂 —— 那是「这道题没有反事实可说」，不是零梯度 bug。
    """
    if scores.ndim != 2 or target.ndim != 1:
        raise ValueError(f'scores {tuple(scores.shape)} / target {tuple(target.shape)} '
                         '应为 [B,K] / [B]')
    if len(kinds) != scores.size(0):
        raise ValueError(f'kinds 有 {len(kinds)} 项，scores 有 {scores.size(0)} 行')
    if accept_mask is not None and accept_mask.shape != scores.shape:
        raise ValueError(f'accept_mask {tuple(accept_mask.shape)} != scores {tuple(scores.shape)}')
    G = int(n_samples)
    if G < 2:
        raise ValueError(f'n_samples={G} < 2 —— 组内归一化至少要两个样本才有 σ')
    if corr_cand not in ('cross', 'same'):
        raise ValueError(f'corr_cand={corr_cand!r} 应为 cross|same')
    if w_task < 0:
        raise ValueError(f'w_task={w_task} 不能为负 —— r̃ 里 r^task 是奖励，翻号就变成罚正确')
    if w_task == 0 and lam_rank <= 0 and lam_corr <= 0:
        raise ValueError('w_task=0 且 λ_rank=λ_corr=0 ⇒ r̃≡0，pg 零梯度；'
                         '要纯整形就把 λ_corr（或 λ_rank）设成非零')
    if beta > 0 and ref_scores is None:
        raise ValueError('beta>0 要 ref_scores（参考策略的逐候选分数）；不要锚就把 beta 设 0')
    if ref_scores is not None and ref_scores.shape != scores.shape:
        raise ValueError(f'ref_scores {tuple(ref_scores.shape)} != scores {tuple(scores.shape)}')
    M = int(n_views)
    if M:
        if M < 2:
            raise ValueError(f'n_views={M} < 2 —— 单视图没有可比的两端')
        if scores.size(0) % M:
            raise ValueError(f'scores 的行数 {scores.size(0)} 不是 n_views={M} 的整数倍')
        if view_kind is None or len(view_kind) != scores.size(0):
            raise ValueError('给了 n_views 就要给等长的 view_kind')
        from .views import VIEW_FULL
        B = scores.size(0) // M
        if any(view_kind[g * M] != VIEW_FULL for g in range(B)):
            raise ValueError('槽 0 必须是 full 视图 —— r^corr 以它为干净支路')
        for g in range(B):
            ks = kinds[g * M:(g + 1) * M]
            if len(set(ks)) != 1:
                raise ValueError(f'组 {g} 的 {M} 个视图题型不一致 {ks} —— 组主序被破坏了')
        if target.view(B, M).ne(target.view(B, M)[:, :1]).any():
            raise ValueError('同一组的 M 个视图 target 不一致 —— 视图只改证据，不改答案')
    elif lam_corr > 0:
        raise ValueError(f'lam_corr={lam_corr} > 0 要多视图批次（--racd-views>=2）；'
                         '单视图臂请设 --grpo-lam-corr 0')
    else:
        B = scores.size(0)

    if view_weight not in ('uniform', 'grounded'):
        raise ValueError(f'view_weight={view_weight!r} 应为 uniform|grounded')
    if view_weight == 'grounded':
        if not M:
            raise ValueError('view_weight=grounded 要多视图批次（--racd-views>=3 + --racd-blind）')

    task_keep = None
    if task_views is not None:
        if task_views not in ('all', 'no_target', 'non_gated'):
            raise ValueError(f'task_views={task_views!r} 应为 all/no_target/non_gated')
        if not M or lam_corr <= 0:
            raise ValueError(f'task_views={task_views!r} 只在「多视图 + lam_corr>0」下有意义：'
                             '它修的是 CE 与 r^corr 在同一 logit 上符号相反，没有 r^corr '
                             '就没有冲突，摘掉反事实视图只是白扔监督')
        if task_views != 'all':
            task_keep = torch.ones(B * M, dtype=torch.bool, device=scores.device)

    if view_weight == 'grounded':
        from .views import VIEW_BLIND
        tk = torch.tensor([vk != VIEW_BLIND for vk in view_kind],
                          dtype=torch.bool, device=scores.device)
        task_keep = tk if task_keep is None else (task_keep & tk)

    logp = _logp(scores)
    p = logp.exp()
    valid = torch.isfinite(scores)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError('有样本一个有效候选都没有 —— 整行被长答案剔除逻辑剔掉了')
    z = scores.new_zeros(())
    kmask = {name: torch.tensor([k == name for k in kinds], dtype=torch.bool,
                                device=scores.device) for name in GRPO_KINDS}

    vw_row, vw_diag = None, {}

    with torch.no_grad():
        pf = p.float().clamp(min=0)
        act = torch.multinomial(pf, G, replacement=True)
        rt = _reward(scores, act, target, kinds, accept_mask)
        conf = pf.gather(1, act)
        rr, viol_rank, rgap_vals = (_rank_shaping(rt, conf, m_rank) if lam_rank > 0
                                    else (torch.zeros_like(rt), z, None))
        rc = torch.zeros_like(rt)
        viol_corr = gate_frac = z
        n_corr = 0
        gap_vals = None
        kind_corr = {name: (0, z, z) for name in GRPO_KINDS}
        if lam_corr > 0:
            from .views import VIEW_TARGET
            m_cpu = _slot_of(view_kind, B, M, VIEW_TARGET)
            rows = [g for g in range(B) if m_cpu[g] > 0]
            n_corr = len(rows)
            if rows:
                gi = torch.tensor(rows, dtype=torch.long, device=scores.device)
                mi = torch.tensor([m_cpu[g] for g in rows], dtype=torch.long,
                                  device=scores.device)
                c3, r3 = conf.view(B, M, G), rt.view(B, M, G)
                c0, c1 = c3[gi, 0], c3[gi, mi]
                if corr_cand == 'same':
                    k = act[gi * M]
                    c1 = pf.view(B, M, -1)[gi, mi].gather(1, k)
                r0, r1 = r3[gi, 0], r3[gi, mi]
                gate = (r0 > r1).to(torch.float32) if gate_corr else torch.ones_like(r0)
                if task_keep is not None:
                    sel = (gate > 0).any(dim=1) if task_views == 'non_gated' \
                        else torch.ones_like(gate[:, 0], dtype=torch.bool)
                    task_keep[(gi * M + mi)[sel]] = False
                ell = F.relu(c1 - c0 + m_corr + alpha_sev * float(sev))
                gated = ell * gate
                rc.view(B, M, G)[gi, 0] = -gated
                rc.view(B, M, G)[gi, mi] = -gated
                ng = gate.sum().clamp(min=1.0)
                gate_frac = gate.mean()
                viol_corr = ((ell > 0).to(torch.float32) * gate).sum() / ng
                gap = c0 - c1
                gap_vals = torch.stack(
                    [gate.sum()]
                    + [((gap < _t).to(torch.float32) * gate).sum() / ng
                       for _t in GRPO_GAP_TH]
                    + [(gap * gate).sum() / ng])
                for name in GRPO_KINDS:
                    sel = kmask[name][gi]
                    nk = int(sel.sum())
                    if nk:
                        gk = gate[sel]
                        kind_corr[name] = (
                            nk, gk.mean(),
                            ((ell > 0).to(torch.float32) * gate)[sel].sum()
                            / gk.sum().clamp(min=1.0))

        r = w_task * rt + lam_rank * rr + lam_corr * rc
        mu = r.mean(dim=1, keepdim=True)
        sd = r.std(dim=1, unbiased=False, keepdim=True)
        adv = (r - mu) / sd.clamp(min=GRPO_ADV_EPS)
        nz = (sd.squeeze(1) > 0).to(torch.float32)

        if view_weight == 'grounded':
            from .views import VIEW_BLIND
            rbar = rt.view(B, M, G).mean(dim=2)
            w = torch.zeros(B, M, device=scores.device, dtype=torch.float32)
            nblind = ground_groups = 0
            for g in range(B):
                slots = view_kind[g * M:(g + 1) * M]
                mb = next((m for m in range(M) if slots[m] == VIEW_BLIND), None)
                if mb is None:
                    w[g] = 1.0 / M
                    continue
                nblind += 1
                margin = (rbar[g] - rbar[g, mb]).clamp(min=0.0)
                margin[mb] = 0.0
                if float(margin.sum()) > 0:
                    ground_groups += 1
                    w[g] = margin / margin.sum()
                else:
                    keep = torch.ones(M, device=w.device)
                    keep[mb] = 0.0
                    w[g] = keep / keep.sum()
            vw_row = w.reshape(B * M)
            vw_diag = {'vw_ground_frac': ground_groups / max(B, 1),
                       'vw_full': float(w[:, 0].mean()),
                       'vw_nblind': float(nblind)}

    if view_weight == 'grounded':
        pg_row = -(logp.gather(1, act) * adv.to(logp.dtype)).mean(dim=1)
        vw = vw_row.to(logp.dtype)
        pg = (vw * pg_row).sum() / vw.sum().clamp(min=1e-6)
    else:
        pg = -(logp.gather(1, act) * adv.to(logp.dtype)).mean()

    if beta > 0:
        ref_logp = _logp(ref_scores.detach().float())
        pm = torch.where(valid, p, torch.zeros_like(p))
        dl = torch.where(valid, logp - ref_logp.to(logp.dtype), torch.zeros_like(logp))
        kl = (pm * dl).sum(dim=-1).mean()
    else:
        kl = z

    task = combined_loss(scores, target, kinds, accept_mask, row_keep=task_keep)
    loss = lam_task * task + pg + beta * kl

    with torch.no_grad():
        vals = torch.stack([task.detach(), pg.detach(), kl.detach(),
                            rt.mean(), rr.mean(), rc.mean(), r.mean(), sd.mean(),
                            nz.mean(), p.max(dim=-1).values.float().mean(),
                            (p.argmax(dim=-1) == target).to(torch.float32).mean(),
                            viol_rank.to(torch.float32), viol_corr.to(torch.float32),
                            gate_frac.to(torch.float32)])
        kind_diag, n_corr_kind = {}, {}
        for name in GRPO_KINDS:
            m = kmask[name]
            n = int(m.sum())
            kind_diag[f'{name}_share'] = n / max(1, scores.size(0))
            kind_diag[f'n_row_{name}'] = n
            if n:
                kind_diag[f'{name}_r_task'] = rt[m].mean()
                kind_diag[f'{name}_adv_nz'] = nz[m].mean()
                kind_diag[f'{name}_hit'] = (p[m].argmax(dim=-1) == target[m]).to(
                    torch.float32).mean()
                kind_diag[f'{name}_pi_max'] = p[m].max(dim=-1).values.float().mean()
            else:
                for s in ('r_task', 'adv_nz', 'hit', 'pi_max'):
                    kind_diag[f'{name}_{s}'] = z
            nk, gk, vk = kind_corr[name]
            kind_diag[f'{name}_gate_frac'] = gk
            kind_diag[f'{name}_viol_corr'] = vk
            n_corr_kind[f'n_corr_{name}'] = nk
    diag = {k: round(v, 6) for k, v in zip(GRPO_DIAG_BASE, vals.tolist())}
    diag.update({k: round(float(v), 6) for k, v in kind_diag.items()})
    diag.update({k: round(float(v), 6) for k, v in vw_diag.items()})
    diag['n_corr'] = n_corr
    diag.update(n_corr_kind)
    if gap_vals is not None:
        gv = gap_vals.tolist()
        diag['n_gate'] = round(gv[0], 4)
        diag.update({gk: round(v, 6) for gk, v in zip(GRPO_GAP_KEYS, gv[1:-1])})
        diag['gap_mean'] = round(gv[-1], 6)
    if rgap_vals is not None:
        rv = rgap_vals.tolist()
        if rv[0] > 0:
            diag['n_pair'] = round(rv[0], 4)
            diag.update({k: round(v, 6) for k, v in zip(GRPO_RGAP_KEYS, rv[1:-1])})
            diag['rgap_mean'] = round(rv[-1], 6)
    return loss, diag


def evgrpo_loss(acq_logits, subsets, present, scores, target, kinds, accept_mask=None,
                *, ref_scores=None, route_mask=None, n_samples=EV_SAMPLES,
                lam_cost=EV_LAM_COST, beta_cal=EV_BETA_CAL, beta_kl=EV_BETA_KL,
                lam_task=EV_LAM_TASK):
    """armEV 证据获取 GRPO -> `(loss, diag)`。

    与 `grpo_loss` 的范式差：动作不是答案候选而是**证据子集**。形状约定（行序 = 题主序，
    行 `b·G+g` = 题 b 的第 g 个子集）：
    * `acq_logits [B, n_slots]`：获取头打分（n_slots=n_organs+1，末位整图），**带梯度**；
    * `subsets [B, G, n_slots]`：Bernoulli 采到的 0/1 子集（动作，detach）；
    * `present [B, n_slots]` bool：哪些槽可获取（在场器官 + 整图），不在场的不进 logπ；
    * `scores [B·G, K]`：每个 masked 子集下答案头的逐候选分数（无效候选 -inf）；
    * `target [B]` / `kinds` 长 B / `accept_mask [B,K]`：逐**题**（G 个子集同题同答案）。

    奖励 `r = correctness − λ_cost·|S| − β_cal·Brier`（correctness=_reward(argmax)，
    Brier=Σ(p−1[gold])²），按题内 G 归一化得优势；`pg=−E[adv·logπ(S)]` 只回传获取头。
    答案头由 `lam_task·combined_loss`（逐题型 proper score，含校准）微调，`beta_kl·KL(ref)`
    弱锚。两个 Brier 角色：奖励里的塑造获取，combined_loss 里的训练答案头。
    """
    if scores.ndim != 2 or target.ndim != 1:
        raise ValueError(f'scores {tuple(scores.shape)} / target {tuple(target.shape)} '
                         '应为 [B·G,K] / [B]')
    B = len(kinds)
    G = int(n_samples)
    if G < 2:
        raise ValueError(f'n_samples={G} < 2 —— 组内归一化至少要两个子集才有 σ')
    if target.shape[0] != B:
        raise ValueError(f'target 有 {target.shape[0]} 项，kinds 有 {B} 项')
    N = B * G
    if scores.shape[0] != N:
        raise ValueError(f'scores 有 {scores.shape[0]} 行，应为 B·G={N}（题主序）')
    n_slots = acq_logits.shape[-1]
    n_organs = n_slots - 1
    if acq_logits.shape != (B, n_slots):
        raise ValueError(f'acq_logits {tuple(acq_logits.shape)} 应为 [B={B}, n_slots={n_slots}]')
    if subsets.shape != (B, G, n_slots):
        raise ValueError(f'subsets {tuple(subsets.shape)} 应为 [B,G,n_slots]=({B},{G},{n_slots})')
    if present.shape != (B, n_slots):
        raise ValueError(f'present {tuple(present.shape)} 应为 [B,n_slots]=({B},{n_slots})')
    if accept_mask is not None and accept_mask.shape != (B, scores.shape[1]):
        raise ValueError(f'accept_mask {tuple(accept_mask.shape)} != [B,K]')
    if beta_kl > 0 and ref_scores is None:
        raise ValueError('beta_kl>0 要 ref_scores（参考策略在**同一 masked 批**上的分数）')
    if ref_scores is not None and ref_scores.shape != scores.shape:
        raise ValueError(f'ref_scores {tuple(ref_scores.shape)} != scores {tuple(scores.shape)}')

    target_r = target.repeat_interleave(G)
    kinds_r = [k for k in kinds for _ in range(G)]
    accept_r = (accept_mask.repeat_interleave(G, dim=0)
                if accept_mask is not None else None)

    logp = _logp(scores)
    p = logp.exp()
    valid = torch.isfinite(scores)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError('有子集一个有效候选都没有 —— 整行被长答案剔除逻辑剔掉了')

    with torch.no_grad():
        act = scores.argmax(dim=-1)
        correctness = _reward(scores, act.unsqueeze(1), target_r, kinds_r,
                              accept_r).squeeze(1)
        onehot = F.one_hot(target_r, scores.size(-1)).to(p.dtype)
        brier = ((p - onehot) ** 2).sum(dim=-1)
        cost = subsets.sum(dim=-1).reshape(N).to(scores.dtype)
        r = correctness - lam_cost * cost - beta_cal * brier
        r2 = r.view(B, G)
        mu = r2.mean(dim=1, keepdim=True)
        sd = r2.std(dim=1, unbiased=False, keepdim=True)
        adv = (r2 - mu) / sd.clamp(min=GRPO_ADV_EPS)
        nz = (sd.squeeze(1) > 0).to(torch.float32)

    logsig_l = -F.softplus(-acq_logits)
    logsig_negl = -F.softplus(acq_logits)
    s = subsets.to(acq_logits.dtype)
    lp = s * logsig_l.unsqueeze(1) + (1 - s) * logsig_negl.unsqueeze(1)
    lp = lp * present.unsqueeze(1).to(acq_logits.dtype)
    logpi = lp.sum(dim=-1)
    pg = -(adv.to(logpi.dtype) * logpi).mean()

    task = combined_loss(scores, target_r, kinds_r, accept_r)
    if beta_kl > 0:
        ref_logp = _logp(ref_scores.detach().float())
        pm = torch.where(valid, p, torch.zeros_like(p))
        dl = torch.where(valid, logp - ref_logp.to(logp.dtype), torch.zeros_like(logp))
        kl = (pm * dl).sum(dim=-1).mean()
    else:
        kl = scores.new_zeros(())
    loss = pg + lam_task * task + beta_kl * kl

    with torch.no_grad():
        hit_row = (p.argmax(dim=-1) == target_r).to(torch.float32)
        pi_row = p.max(dim=-1).values.float()
        whole_frac = subsets[:, :, -1].to(torch.float32).mean()
        if route_mask is not None:
            acq_org = subsets[:, :, :n_organs].to(torch.float32)
            bound = route_mask[:, None, :n_organs].to(torch.float32)
            bound_frac = ((acq_org * bound).sum()
                          / acq_org.sum().clamp(min=1.0))
        else:
            bound_frac = scores.new_zeros(())
        kmask = {name: torch.tensor([k == name for k in kinds], dtype=torch.bool,
                                    device=scores.device) for name in GRPO_KINDS}
        vals = torch.stack([task.detach(), pg.detach(), kl.detach(),
                            cost.mean(), brier.mean(), correctness.mean(),
                            nz.mean(), bound_frac, whole_frac,
                            hit_row.mean(), pi_row.mean(), r.mean(), sd.mean()])
        kind_diag = {}
        for name in GRPO_KINDS:
            m = kmask[name]
            nq = int(m.sum())
            kind_diag[f'{name}_share'] = nq / max(1, B)
            kind_diag[f'n_row_{name}'] = nq
            mr = m.repeat_interleave(G)
            if nq:
                kind_diag[f'{name}_hit'] = hit_row[mr].mean()
                kind_diag[f'{name}_pi_max'] = pi_row[mr].mean()
                kind_diag[f'{name}_adv_nz'] = nz[m].mean()
            else:
                for sfx in ('hit', 'pi_max', 'adv_nz'):
                    kind_diag[f'{name}_{sfx}'] = scores.new_zeros(())
    diag = {k: round(v, 6) for k, v in zip(EVGRPO_DIAG_BASE, vals.tolist())}
    diag.update({k: round(float(v), 6) for k, v in kind_diag.items()})
    return loss, diag


DPO_BETA = 0.1
DPO_LAM_CE = 0.0
DPO_DIAG = ('dpo', 'margin', 'gate_frac', 'q_full', 'q_blind', 'ce')
DPO_DIAG_W = dict.fromkeys(DPO_DIAG)
DPO_DIAG_W.update({'dpo': 'n_pairs', 'margin': 'n_pairs'})


def dpo_loss(scores, target, kinds, view_kind, *, n_views, ref_scores,
             beta=DPO_BETA, lam_ce=DPO_LAM_CE, accept_mask=None):
    """-> `(loss, diag)`；跨视图先验感知硬负例 DPO（见上方段注）。

    多视图批次（组主序：行 r 属组 r//M、槽 r%M，槽 0 恒 full），需 blind 槽提供先验答案。
    `scores [B*M,K]`（无效候选 -inf），`ref_scores` 同形（参考策略逐候选分数，beta>0 必需）。
    梯度只走 full 视图的 logπ(gold) 与 logπ(prior)；blind 视图只供 argmax 标签（无梯度）。
    """
    from .views import VIEW_FULL, VIEW_BLIND
    if scores.ndim != 2 or target.ndim != 1:
        raise ValueError(f'scores {tuple(scores.shape)} / target {tuple(target.shape)} '
                         '应为 [B*M,K] / [B*M]')
    if len(kinds) != scores.size(0):
        raise ValueError(f'kinds 有 {len(kinds)} 项，scores 有 {scores.size(0)} 行')
    M = int(n_views)
    if M < 2:
        raise ValueError(f'n_views={M} < 2 —— dpo 要 full + blind 两端（--racd-views>=2）')
    if scores.size(0) % M:
        raise ValueError(f'scores 行数 {scores.size(0)} 不是 n_views={M} 的整数倍')
    if view_kind is None or len(view_kind) != scores.size(0):
        raise ValueError('给了 n_views 就要给等长的 view_kind')
    B = scores.size(0) // M
    if any(view_kind[g * M] != VIEW_FULL for g in range(B)):
        raise ValueError('槽 0 必须是 full 视图 —— dpo 在证据视图上取 chosen/rejected 的 logπ')
    for g in range(B):
        ks = kinds[g * M:(g + 1) * M]
        if len(set(ks)) != 1:
            raise ValueError(f'组 {g} 的 {M} 个视图题型不一致 {ks} —— 组主序被破坏了')
    if target.view(B, M).ne(target.view(B, M)[:, :1]).any():
        raise ValueError('同一组的 M 个视图 target 不一致 —— 视图只改证据，不改答案')
    if beta <= 0:
        raise ValueError(f'beta={beta} <= 0 —— dpo 要参考锚（beta>0）；不要锚请换损失')
    if ref_scores is None:
        raise ValueError('dpo 要 ref_scores（参考策略逐候选分数）')
    if ref_scores.shape != scores.shape:
        raise ValueError(f'ref_scores {tuple(ref_scores.shape)} != scores {tuple(scores.shape)}')

    s = scores.float()
    logp = _logp(s)
    ref_logp = _logp(ref_scores.detach().float())
    valid = torch.isfinite(s)
    rows_f = torch.arange(B, device=s.device) * M
    lp_f = logp[rows_f]
    rlp_f = ref_logp[rows_f]
    gold = target[rows_f]

    b_slot = _slot_of(view_kind, B, M, VIEW_BLIND)
    has_b = torch.tensor([b >= 0 for b in b_slot], dtype=torch.bool, device=s.device)
    b_rows = torch.tensor([g * M + (b_slot[g] if b_slot[g] >= 0 else 0) for g in range(B)],
                          dtype=torch.long, device=s.device)
    prior = logp[b_rows].argmax(dim=-1)
    pair = has_b & (prior != gold)
    pair = pair & valid[rows_f].gather(1, gold.unsqueeze(1)).squeeze(1) \
                & valid[rows_f].gather(1, prior.unsqueeze(1)).squeeze(1)

    lr_gold = (lp_f.gather(1, gold.unsqueeze(1)).squeeze(1)
               - rlp_f.gather(1, gold.unsqueeze(1)).squeeze(1))
    lr_prior = (lp_f.gather(1, prior.unsqueeze(1)).squeeze(1)
                - rlp_f.gather(1, prior.unsqueeze(1)).squeeze(1))
    margin = lr_gold - lr_prior
    dpo_row = -F.logsigmoid(beta * margin)
    pf = pair.to(s.dtype)
    denom = pf.sum().clamp(min=1.0)
    dpo = (dpo_row * pf).sum() / denom
    npair = int(pair.sum())
    margin_mean = (margin.detach() * pf).sum() / denom

    if lam_ce > 0:
        keep = (torch.arange(B * M, device=s.device) % M == 0)
        ce = combined_loss(scores, target, kinds, accept_mask, row_keep=keep)
    else:
        ce = s.new_zeros(())
    loss = dpo + lam_ce * ce

    with torch.no_grad():
        q_full = (lp_f.argmax(dim=-1) == gold).to(torch.float32).mean()
        q_blind = ((prior[has_b] == gold[has_b]).to(torch.float32).mean()
                   if bool(has_b.any()) else s.new_zeros(()))
        gate_frac = pair.to(torch.float32).mean()
        vals = torch.stack([dpo.detach(), margin_mean.detach(), gate_frac.detach(),
                            q_full.detach(), q_blind.detach(), ce.detach()])
    diag = {k: round(v, 6) for k, v in zip(DPO_DIAG, vals.tolist())}
    diag['n_pairs'] = npair
    return loss, diag
