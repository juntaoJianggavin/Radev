#!/usr/bin/env python3
"""verdicts_shard*.jsonl → 论文表冻结口径的全部指标。

口径（2026-10-30 与用户对齐，回归锚 = 8 列权威表 + 12 列扩展贴入行）：
  noul   acc    全部 noul 行的微观命中（含 newq/triage）。
         mBAcc  30 种 finding 病的宏平均，每病 = 类平衡召回 (TPR+TNR)/2；
                parsed=None（不可解析）的行剔除 —— gen 读出下不可解析不算错、不进分母。
         mAUC   同 30 病，每病 AUROC(P(Yes))，P 取全局 Yes 标签（raw 键）。
         mBrier 同 30 病，每病 mean((P(Yes) − gold)²) 的宏平均（2026-11-03 用户指令
                「把 bacc 都换成 brier」，表里替掉 mBAcc 列；mBAcc 仍算，供指纹与对账）。
                行集与 mAUC 一致（有 probs 的行），**不剔** parsed=None。gen 无 probs ⇒
                None ⇒ 表里 “--”。地板**逐病不同** = prev_d(1−prev_d)（30 病 prev 从
                0.027 铺到 0.976 ⇒ 地板 0.023–0.250，10.7× 差），所以这一列没有 mBAcc
                那种统一的 0.5000 锚；恒答 Yes 的病级 Brier = 1−prev_d、恒答 No = prev_d，
                两者都比气候地板高。要统一地板得除成 BSS = 1 − Brier/气候，本次未做。
         mF1    同 30 病，每病 F1(正类=Yes)=2TP/(2TP+FP+FN) 的宏平均（2026-11-03 用户
                指令：noul/choice 两支都加）。与 mBAcc 同分母（parsed=None 剔除）。
                **注意 F1 没有 BAcc 那种固定地板**：恒答阳性的逐病 F1 = 2p/(1+p)，随
                prevalence 严格单调，而 30 病 p 从 0.027 铺到 0.976 ⇒ 实测宏平均地板
                always-Yes = 0.5694、always-No = 0.0000（两者 mBAcc 都是 0.5000）。
                读数必须连着这两个地板一起看，不能当「越高越好、0.5 是瞎猜」用。
  choice acc    全部 choice 行微观命中。
         mBAcc  26 种病（organ_* 题选项里出现的疾病，去掉 no_abnormality）的宏平均，
                每病池化所有选项含该病的题，(该病召回 + 非该病召回)/2；
                不可解析算错（2026-10-30 用户口径：choice 不可解析都算错）。
         mAUC   同 26 病，每病池化 one-vs-rest AUROC(P(病))。
         mBrier 同 26 病，每病 one-vs-rest mean((P(该病) − 1[gold=该病])²) 的宏平均
                （同日上表替掉 choice mBAcc 列）。**用二分类 OvR 而不是多类 Brier**：
                多类的均匀地板 (K−1)/K 随选项数 K 变（实测本集 K∈{2,3,4,5,6,10,11}，
                地板 0.500→0.909），会把选项长度异质性整个搬进宏平均；OvR 下 25/26 病
                池内 K 唯一（每病只落在 1 个 qid、qid 选项表跨 study 恒定，实测 0/93 变），
                唯一例外 submucosal_edema 混 5选1646/6选822（prev 0.1531/0.1180），
                占全部（病,行）对的 1.89%。地板同样逐病不同 = prev_d(1−prev_d)，
                choice 侧 prev 实测 0.0015（aortic_valve_calcification，3 正/1974 负）
                到 0.6363（atelectasis，1258 正）⇒ 地板 0.0015–0.2426，跨病差 160×，
                比 noul 侧更不能当「统一锚」读。
         mF1    同 26 病，每病 one-vs-rest F1(正类=该病) 的宏平均。不可解析算错 ⇒ pos 组
                的不可解析行是 FN；neg 组的不可解析行既非 FP 也非 TN，而 F1 不看 TN ⇒
                那条口径在 neg 侧不进 F1（choice mBAcc 的 TNR 项则全部由它驱动）。
                地板是逐病包络（每个病各取「恒选该病」，不是单个可实现的模型）：
                实测宏平均 = 0.2346，mBAcc 同样是 0.5000。全表 34 个读出里有 30 个
                低于这条地板 ⇒ 这一列必须连着地板一起读。
  score  MAE/R² 概率加权期望档位 E；无 probs（gen）退化为解析档位；
                不可解析不进计算（score 不可解析不计算）。
         QWK    argmax 档位（gen 用解析档位）的微观二次加权 κ，×100；
                不可解析同样不进计算。

  ── 校准块（2026-11-05 新增，#190；RL 臂判据②「校准没变差」的前置）─────────────
  每支一套 `cal_*`（支级、行集=该行有完整 probs 的行，`cal_n_cal` 报分母），外加
  noul/choice 的逐病宏平均 `mBSS`/`mECE`（与 mBrier/mAUC 同一行集、同一批病）。
    mBSS   逐病 1 − Brier_d/气候_d，气候_d = 本病经验边际的 prev(1−prev)。把 mBrier
           除不掉的「30 病地板差 10.7×、choice 侧 160×」除掉；0 = 与不看图像的边际
           猜测同分，<0 = 比边际猜测还差。恒预测边际的自检恰为 0。
    mECE   逐病二分类可靠性误差（P(正类) vs gold，10 等宽箱，按箱计数加权）再宏平均。
    cal_Brier / cal_BSS   **多类** Brier（本支候选集上 Σ(p−onehot)²）。K 不唯一时
           （choice K∈{2..11}、score 档数不止一种）按 K 分桶、各算气候再按行数加权
           合并 —— 跨 K 直接池化就是把选项数异质性搬进读数（与上面拒绝多类宏平均同
           一条理由）。K=2 时恒有 cal_RPS = cal_Brier/2 且 RPSS = BSS（自检）。
    cal_RPS / cal_RPSS    KRPS（除 K−1，与 MAE 同量纲级），只在**有序**支出：
           score 有、noul 有（K=2 时逐字等于二分类 Brier）、choice **无**（候选是无序
           疾病列表，不硬排队列出一个假数）。
    cal_topECE            支级 top-label ECE，置信度 = 原始 p_max，10 等宽箱，空箱不补。
           与 analysis/confidence_curve.py 的 Jev 官方 confidence（(p_max−1/n)/(1−1/n)）
           **不同尺度，不可互比**。
    cal_AURC              覆盖率–准确率曲线下面积（越大越好）。**必须与同支 acc 连着读**：
           AURC−acc 才是拒答净增益；常数置信度的退化情形按块内随机次序的期望计入，
           此时 AURC 恰等于 acc（这条列的自检，别让并列被排序键排成「先对后错」）。
    correct 一律取冻结的 ok（= 表里 acc 的同源判定；score 那条就是 argmax 档==gold，
           与 QWK 同源）。gen 臂无 probs ⇒ 整块不进（不静默按 0 或按 0.5 补）。
    注：支级 cal_* 覆盖**全部**该支行（含 newq_*/triage_*），而 mBrier/mBSS/mECE 只
    覆盖 30/26 病的宏池 —— 两列的分母不同，读数别串。
    并排看几个 run：python3 analysis/verdict_metrics.py --cmp <dir>:标签 <dir>:标签 …

    🔴 顺带修的口径 bug（#187，choice 疾病级 mAUC/mBrier）：旧实现从**第一个**含该病的
    qid 的第一行嗅一个列名 pk 全场共用，且 `for r0` 的 `break` 落在 `if` 外面（首行无
    probs ⇒ 整病静默丢）。现在按**每行自己的 qid** 的候选序定列。实测 s32 基座：26 病里
    25 病逐字节不变，只有 submucosal_edema（跨 12 个 qid、位次不同）被纠正，
    ΔmAUC=+0.164、ΔmBrier=+0.022 ⇒ 宏平均 choice mAUC **+0.0063**、mBrier +0.0009
    （旧值偏乐观）。全表所有 run 同向受影响，重不重出权威表是单独的决定。
  overall acc   noul+choice 微观命中（score 不进 acc；不可解析算错）。
         mBAcc  30 种 finding 病，逐病取两支的**等权平均**再宏平均（2026-11-03 口径）：
                b(d) = ½·(b_noul(d) + b_choice(d))；某病只有一支有分母就只用那一支
                （lymphadenopathy / pancreatic_atrophy / prostatomegaly / splenomegaly
                四病无任何 choice 行）。任何恒答常数的零信息模型此格恒 = 50.00。
         mAUC   **不出**。两支分数尺度不可比（noul 是 P(Yes)、choice 是 P(该病)），
                池化进同一个 ranking 无法解释，重加权也救不回来；要看 AUC 分别看
                noul mAUC 与 choice mAUC 两列。

    为什么 overall mBAcc 不能按行级池化（2026-11-02 那版，已废）：池化后
    TPR = wp·TPR_noul + (1-wp)·TPR_choice、TNR = wn·TNR_noul + (1-wn)·TNR_choice，
    wp = P_n/(P_n+P_c)、wn = N_n/(N_n+N_c)，只有 wp == wn 才等于两支等权混合。
    实测 30 病 wp ∈ [0.50,1.00] 而 wn ∈ [0.006,1.00]，几乎处处不等：每个病的阳性
    与阴性各由不同题型主导（aortic_valve_calcification：P_n=29/N_n=25 对
    P_c=3/N_c=1974 ⇒ wp=0.906、wn=0.013），于是「noul 恒答 Yes + choice 恒选非该病」
    这个不读图的常数模型在该病拿 0.947，宏平均全表 = 65.91（逐病挑最优常数 75.52），
    而同一模型在 noul 单列 / choice 单列 / 旧的 (30·noul+26·choice)/56 上都正好 50.00。
    这就是当时「overall 比两支都高」的全部来源。

    python3 analysis/verdict_metrics.py <verdicts_dir> <input.jsonl>
    python3 analysis/verdict_metrics.py <verdicts_dir> <input.jsonl> --noul-dir <dir>
        拼接线用：noul 行取 --noul-dir（新 prompt 重跑），choice/score 取主目录。
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

LETTERS = "abcdefghijklmnopqrstuvwxyz"
_INPUT_CACHE = {}


def norm(k):
    return str(k).removeprefix("Answer: ").strip()


def auroc(pos, neg):
    if not pos or not neg:
        return None
    xs = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1][0] == xs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    s_pos = sum(r for r, (_, y) in zip(ranks, xs) if y == 1)
    n_pos, n_neg = len(pos), len(neg)
    return (s_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_mc(pv, g):
    """多类 Brier = Σ_k (p_k − 1[k=g])²。K=2 时逐字等于二分类 Brier (p−y)²。"""
    return sum((p - (1.0 if i == g else 0.0)) ** 2 for i, p in enumerate(pv))


def rps(pv, g):
    """排序概率得分 KRPS = 1/(K−1)·Σ_{j<K}(F_j − G_j)²，F_j=Σ_{i≤j}p_i、G_j=1[g≤j]。

    G 是真值 one-hot 的**累积**：档位 ≤j 就等于 1（不是「j≤g」，写反会把好预测判成
    最差 —— 自检：K=2 时本式与二分类 Brier 逐字相等，且恰为多类 Brier 的一半）。
    只有候选**有序**才有意义：score（档位）有，noul（No<Yes）有且 K=2 时退化成
    Brier（代数恒等，所以 noul 不单列 RPS），choice（同一器官的疾病列表）没有序
    ⇒ 那一支的 RPS 不出，不用「按字母序硬排」糊一个数出来。
    """
    k = len(pv)
    if k < 2 or g is None or not (0 <= g < k):
        return None
    s = cp = cg = 0.0
    for j in range(k - 1):
        cp += pv[j]
        cg = 1.0 if g <= j else 0.0
        s += (cp - cg) ** 2
    return s / (k - 1)


def ece_top(pairs, nb=10):
    """top-label ECE：pairs=[(p_max, correct)]，等宽 nb 箱，Σ_b (n_b/N)·|acc_b − conf_b|。

    空箱不进（不补 0，否则等于把「没有这一箱」当成「校准完美」）。置信度用**原始
    p_max**，不是 analysis/confidence_curve.py 那套 Jev 官方 confidence
    （(p_max−1/n)/(1−1/n) 削基线）—— 两套数字不同尺度，不可互比；这里选 p_max 是为了
    和文献的 ECE/AURC 同尺度。
    """
    if not pairs:
        return None
    b = defaultdict(lambda: [0, 0.0, 0.0])
    for c, ok in pairs:
        i = min(int(c * nb), nb - 1)
        e = b[i]
        e[0] += 1
        e[1] += c
        e[2] += 1.0 if ok else 0.0
    n = len(pairs)
    return sum(v[0] / n * abs(v[2] / v[0] - v[1] / v[0]) for v in b.values())


def ece_binary(pairs, nb=10):
    """二分类可靠性误差（Brier 的分解视图）：pairs=[(P(正类), y)]，等宽 nb 箱。"""
    if not pairs:
        return None
    b = defaultdict(lambda: [0, 0.0, 0.0])
    for p, y in pairs:
        i = min(int(p * nb), nb - 1)
        e = b[i]
        e[0] += 1
        e[1] += p
        e[2] += y
    n = len(pairs)
    return sum(v[0] / n * abs(v[2] / v[0] - v[1] / v[0]) for v in b.values())


def aurc(pairs):
    """AURC = ∫_0^1 acc(c)dc（按置信度降序保留前 c 比例的覆盖率–准确率曲线下面积）。

    越大越好；**必须和同支 acc 连着读**：随机置信度 ⇒ AURC ≈ acc（没有可用的拒答
    信号），完美排序 ⇒ 1.0。所以「AURC−acc」才是选择性预测的净增益，单看 AURC 会把
    「本来就答得准」误读成「知道自己哪里答不准」。

    **同置信度的行按块内随机次序的期望计入**：直接对 (−conf, 对错) 排序会让并列行被
    排序键悄悄排成「先对后错」，凭空把 AURC 抬到接近 1 —— 常数置信度的退化情形必须
    恰好等于 acc，那是这条列的自检。实现：置信度降序逐块，块内第 t 行的运行准确率取
    (前面已对 s + t·块内正确率) / k，对块内所有排列是精确期望，无随机种子、可复算。
    """
    if not pairs:
        return None
    bl = defaultdict(lambda: [0, 0])
    for c, ok in pairs:
        e = bl[c]
        e[0] += 1
        e[1] += 1 if ok else 0
    n = len(pairs)
    tot = k = s = 0.0
    for conf in sorted(bl, reverse=True):
        m, bc = bl[conf]
        for t in range(1, int(m) + 1):
            k += 1
            tot += (s + t * bc / m) / k
        s += bc
    return tot / n


def calib_vecs(vecs, ordinal, nb=10):
    """[(概率向量, gold 下标, correct)] -> 该支的校准块（含气候地板）。

    地板 = **在本支同一批行上**回代经验边际分布（= 恒预测先验、不看图像的模型）：
    Brier_clim / RPS_clim，于是 BSS = 1 − Brier/Brier_clim、RPSS = 1 − RPS/RPS_clim
    把「逐病/逐 qid 地板差 10×~160×」这件事除掉了（mBrier 那一列除不掉，见文件头）。
    BSS/RPSS = 0 ⇒ 与不看图像的边际猜测同分；<0 ⇒ 比边际猜测还差。

    **候选数 K 不唯一时按 K 分桶**：choice 的 K∈{2..11}、score 的档数也不止一种，
    多类 Brier/RPS 的量纲随 K 变（均匀地板 (K−1)/K 从 0.500 铺到 0.909），跨 K 直接池化
    就是把选项数异质性搬进读数 —— 与文件头拒绝「多类 Brier 宏平均」同一条理由。
    分桶后各组的 Brier/气候/RPS 按行数加权合并（一个加权平均，不是逐 K 一列），
    topECE/AURC 不看 K（只用 p_max 和 correct），直接在全部行上池化。
    """
    if not vecs:
        return {"n_cal": 0}
    vecs = [(pv, g, c) for pv, g, c in vecs if isinstance(g, int) and 0 <= g < len(pv)]
    if not vecs:
        return {"n_cal": 0}
    pairs = [(max(pv), bool(c)) for pv, _, c in vecs]
    out = {"n_cal": len(vecs),
           "topECE": round(ece_top(pairs, nb), 4),
           "AURC": round(aurc(pairs), 4)}
    ks = sorted({len(pv) for pv, _, _ in vecs})
    out["K"] = ks[0] if len(ks) == 1 else f'{ks[0]}..{ks[-1]}'
    n_b = nb_r = nb_c = nr_r = nr_c = 0.0
    n_used = 0
    for k in ks:
        grp = [(pv, g) for pv, g, _ in vecs if len(pv) == k]
        n = len(grp)
        marg = [0.0] * k
        for _, g in grp:
            if 0 <= g < k:
                marg[g] += 1.0
        marg = [m / n for m in marg]
        nb_r += sum(brier_mc(pv, g) for pv, g in grp)
        nb_c += sum(brier_mc(marg, g) for _, g in grp)
        n_used += n
        if ordinal and k >= 2:
            rp = [rps(pv, g) for pv, g in grp]
            rc = [rps(marg, g) for _, g in grp]
            if all(v is not None for v in rp + rc):
                nr_r += sum(rp)
                nr_c += sum(rc)
    br = nb_r / n_used
    bc = nb_c / n_used
    out["Brier"] = round(br, 4)
    out["Brier_clim"] = round(bc, 4)
    out["BSS"] = round(1 - br / bc, 4) if bc > 0 else None
    if ordinal and nr_r:
        m, mc = nr_r / n_used, nr_c / n_used
        out["RPS"] = round(m, 4)
        out["RPS_clim"] = round(mc, 4)
        out["RPSS"] = round(1 - m / mc, 4) if mc > 0 else None
    elif ordinal:
        out["RPS"] = None
    return out


def qwk(pred, gold, k):
    if k < 2:
        return None
    cm = [[0] * k for _ in range(k)]
    for p, g in zip(pred, gold):
        cm[g][p] += 1
    n = sum(sum(r) for r in cm)
    if not n:
        return None
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    obs = sum(w[g][p] * cm[g][p] for g in range(k) for p in range(k))
    row = [sum(cm[g]) for g in range(k)]
    col = [sum(cm[g][p] for g in range(k)) for p in range(k)]
    exp = sum(w[g][p] * row[g] * col[p] / n for g in range(k) for p in range(k))
    return 1.0 - obs / exp if exp else None


def load_input(input_path):
    """input jsonl → choice qid 的选项表 / score 档位表 / 30 病与 26 病名单。"""
    cached = _INPUT_CACHE.get(str(input_path))
    if cached is not None:
        return cached
    choice_crit, score_levels = {}, {}
    for line in open(input_path, encoding="utf-8"):
        e = json.loads(line)
        q = e["question"]
        qid = e["question_id"]
        if q["type"] == "choice" and qid not in choice_crit:
            choice_crit[qid] = list(q["criteria"].keys())
        elif q["type"] == "score" and qid not in score_levels:
            crit = q["criteria"]
            vals = list(crit.values()) if isinstance(crit, dict) else list(crit)
            score_levels[qid] = [norm(v) for v in vals]
    noul_dis = {q[len("finding_"):] for q in choice_crit if q.startswith("finding_")}
    noul_dis = set()
    choice_dis = set()
    for line in open(input_path, encoding="utf-8"):
        qid = json.loads(line)["question_id"]
        if qid.startswith("finding_"):
            noul_dis.add(qid[len("finding_"):])
    for qid, keys in choice_crit.items():
        if qid.startswith("organ_"):
            choice_dis |= set(keys)
    choice_dis -= {"no_abnormality"}
    if "usable_test" in str(input_path):
        assert len(noul_dis) == 30, len(noul_dis)
        assert len(choice_dis) == 26, len(choice_dis)
    else:
        assert noul_dis, "输入里一个 finding_* 病都没有 —— 喂错文件了？"
        assert choice_dis, "输入里 organ_* choice 选项病为空 —— 喂错文件了？"
    _INPUT_CACHE[str(input_path)] = (choice_crit, score_levels, noul_dis, choice_dis)
    return choice_crit, score_levels, noul_dis, choice_dis


def load_rows(d):
    """读一个产物目录的全部逐题判定行，按 (study, question, type) 去重（保留首现）。

    去重是必需的：断点续跑/多 shard 重投会写出完全相同的重复行（ct_chat 的 words 臂
    有 1,051 行重复 noul，两份逐字节相同）。不去重会把 noul 分母从 37,925 撑到
    38,976，acc/mBAcc 全偏。其余产物目录都恰好 56,825 行，去重是空操作。
    """
    rows, seen = [], set()
    for f in sorted(glob.glob(str(d / "verdicts_shard*.jsonl")) + glob.glob(str(d / "verdicts_spliced.jsonl"))):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = (r.get("study_id"), r.get("question_id"), r.get("type"))
            if k in seen:
                continue
            seen.add(k)
            rows.append(r)
    return rows


def yes_key(rs):
    """noul 行的 P(Yes) 概率键（raw 键 'b'=Yes 的符号约定；短语方言无 'a'/'b' 键）。"""
    for r in rs:
        if r.get("parsed") == 1 and isinstance(r.get("probs"), dict) and r["raw"] in r["probs"]:
            return r["raw"]
    for r in rs:
        p = r.get("probs")
        if isinstance(p, dict) and "a" in p and "b" in p:
            return "b"
    for r in rs:
        p = r.get("probs")
        if not isinstance(p, dict):
            continue
        by = {}
        for k in p:
            by.setdefault(norm(k), k)
        if "a" in by and "b" in by:
            return by["b"]
    return None


def _idx_of(g, keys):
    """gold → 候选下标。gold 可能是候选原文（shared/semantic 空间）或候选字母（符号空间）。"""
    if g in keys:
        return keys.index(g)
    if isinstance(g, str):
        n = norm(g)
        if len(n) == 1 and n.isalpha() and n.lower() in LETTERS:
            i = LETTERS.index(n.lower())
            return i if i < len(keys) else None
    return None


def _cvec(r, keys):
    """choice/score 行的 probs → 与 keys 同序的概率向量；任一键对不上则 None（该行不进校准块）。

    键有三代写法（候选原文 / 小写字母符号 / 大写字母 native 方言），与 choice 侧
    `km.get(dis) or km.get(letter) or km.get(letter.upper())` 同源。**按本行自己的 qid
    的候选序**解析，不借用别的 qid 的字母 —— 这就是 #187 那条：同一病可以出现在多个
    qid 且在该 qid 里位次不同（submucosal_edema 混 5 选/6 选），拿第一个 qid 的字母去
    查别的 qid 的行会静默取错列。写盘时已经是候选集上的 softmax，这里只兜浮点残差。
    """
    p = r.get("probs")
    if not isinstance(p, dict) or not keys:
        return None
    km = {norm(k): k for k in p}
    out_ = []
    for i, k0 in enumerate(keys):
        kk = k0 if k0 in p else km.get(norm(k0))
        if kk is None and i < len(LETTERS):
            a = LETTERS[i]
            kk = km.get(a) or km.get(a.upper())
        v = p.get(kk) if kk is not None else None
        if not isinstance(v, (int, float)):
            return None
        out_.append(float(v))
    t = sum(out_)
    return [x / t for x in out_] if t > 0 else None


def group(rows):
    g = defaultdict(lambda: defaultdict(list))
    for r in rows:
        g[r["type"]][r["question_id"]].append(r)
    return g


def compute(d, inp, noul_dir=None):
    """一个 run（可选拼接 noul 来源）→ 冻结口径指标 dict。main()/table30.py 共用。"""
    d = Path(d)
    noul_dir = Path(noul_dir) if noul_dir else None
    choice_crit, score_levels, noul_dis, choice_dis = load_input(inp)
    by = group(load_rows(d))
    noul_by = group(load_rows(noul_dir))["noul"] if noul_dir else by.get("noul", {})
    rows_n = sum(len(v) for v in noul_by.values())

    out = {"n": rows_n + sum(len(r) for t, qs in by.items() if t != "noul" for r in qs.values())}
    if noul_dir:
        out["noul_dir"] = str(noul_dir)
    letter_of = {qid: {k: LETTERS[i] for i, k in enumerate(keys)}
                 for qid, keys in choice_crit.items()}
    dis_of = {}
    for qid, keys in choice_crit.items():
        lm = {v: k for k, v in letter_of[qid].items()}
        dis_of[qid] = lambda r, keys=keys, lm=lm: r["gold"] if r["gold"] in keys else lm.get(r["gold"])
    qs = noul_by
    rs = [r for q in qs.values() for r in q]
    cell = {"n": len(rs)}
    if rs:
        cell["acc"] = round(sum(1 for r in rs if r.get("ok")) / len(rs), 4)
        yl = yes_key(rs)
        ba, au, f1s, br, bss, ece = [], [], [], [], [], []
        for dis in sorted(noul_dis):
            qr = qs.get("finding_" + dis)
            if not qr:
                continue
            keep = [r for r in qr if r.get("parsed") is not None]
            if keep:
                rec = []
                for c in {r["gold"] for r in keep}:
                    hit = sum(1 for r in keep if r["gold"] == c)
                    rec.append(sum(1 for r in keep if r["gold"] == c and r["parsed"] == c) / hit)
                ba.append(sum(rec) / len(rec))
                tp = sum(1 for r in keep if r["gold"] == 1 and r["parsed"] == 1)
                fp = sum(1 for r in keep if r["gold"] == 0 and r["parsed"] == 1)
                fn = sum(1 for r in keep if r["gold"] == 1 and r["parsed"] != 1)
                if tp + fn:
                    f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            if yl:
                a = auroc([r["probs"][yl] for r in qr if r["gold"] == 1],
                          [r["probs"][yl] for r in qr if r["gold"] == 0])
                if a is not None:
                    au.append(a)
                pb = [(r["probs"][yl], r["gold"]) for r in qr
                      if isinstance(r.get("probs"), dict) and yl in r["probs"]
                      and r["gold"] in (0, 1)]
                if pb:
                    bd = sum((p - g) ** 2 for p, g in pb) / len(pb)
                    br.append(bd)
                    pv_ = sum(g for _, g in pb) / len(pb)
                    cl = pv_ * (1 - pv_)
                    if cl > 0:
                        bss.append(1 - bd / cl)
                    e = ece_binary(pb)
                    if e is not None:
                        ece.append(e)
        cell["mBAcc"] = round(sum(ba) / len(ba), 4) if ba else None
        cell["mAUC"] = round(sum(au) / len(au), 4) if au else None
        cell["mF1"] = round(sum(f1s) / len(f1s), 4) if f1s else None
        cell["mBrier"] = round(sum(br) / len(br), 4) if br else None
        cell["mBSS"] = round(sum(bss) / len(bss), 4) if bss else None
        cell["mECE"] = round(sum(ece) / len(ece), 4) if ece else None
        if yl:
            vv = [([1.0 - r["probs"][yl], r["probs"][yl]], r["gold"], bool(r.get("ok")))
                  for r in rs if isinstance(r.get("probs"), dict)
                  and yl in r["probs"] and r["gold"] in (0, 1)]
            cell.update({"cal_" + k: v for k, v in calib_vecs(vv, ordinal=True).items()})
    out["noul"] = cell

    qs = by.get("choice", {})
    rs = [r for q in qs.values() for r in q]
    cell = {"n": len(rs)}
    if rs:
        cell["acc"] = round(sum(1 for r in rs if r.get("ok")) / len(rs), 4)
        ba, au, f1s, br, bss, ece = [], [], [], [], [], []

        def pcol(r, dis):
            """该行 probs 里 dis 那一列的概率；列位按**本行自己 qid** 的候选序定（#187）。

            None = 该行没有完整可用的分布（未覆盖 / 生成式无 probs / 键对不上），
            不进 AUROC/Brier 的分子分母。旧写法是从第一个 qid 的第一行猜一个列名 pk
            然后全场共用，有两处静默失效：① 那个 `for r0 in qr` 的 `break` 落在 `if`
            外面，首行没有 probs 就整病被丢出 mAUC/mBrier，且不留痕迹；② 同一病落在
            多个 qid 时位次可以不同（submucosal_edema 混 5 选 1646 / 6 选 822），别的
            qid 的行会被查错列。
            """
            keys = choice_crit.get(r["question_id"], [])
            if dis not in keys:
                return None
            v = _cvec(r, keys)
            return None if v is None else v[keys.index(dis)]

        for dis in sorted(choice_dis):
            pool = [r for qid, qr in qs.items() if dis in choice_crit.get(qid, []) for r in qr]
            if not pool:
                continue
            pos = [r for r in pool if dis_of[r["question_id"]](r) == dis]
            neg = [r for r in pool if dis_of[r["question_id"]](r) != dis]
            rec, pds = [], {}
            for grp, hit_dis in ((pos, True), (neg, False)):
                if grp:
                    pd = [_pred_dis(r, choice_crit, letter_of) for r in grp]
                    pds[hit_dis] = list(zip(grp, pd))
                    ok = sum(1 for r, p in zip(grp, pd)
                             if (r.get("parsed") is not None) and ((p == dis) == hit_dis))
                    rec.append(ok / len(grp))
            if rec:
                ba.append(sum(rec) / len(rec))
            tp = sum(1 for r, p in pds.get(True, [])
                     if r.get("parsed") is not None and p == dis)
            fp = sum(1 for r, p in pds.get(False, [])
                     if r.get("parsed") is not None and p == dis)
            if pos:
                fn = len(pos) - tp
                f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            pr = [(pcol(r, dis), 1.0 if dis_of[r["question_id"]](r) == dis else 0.0)
                  for r in pool]
            pr = [(p, g) for p, g in pr if p is not None]
            if pr:
                pos = [p for p, g in pr if g == 1.0]
                neg = [p for p, g in pr if g == 0.0]
                a = auroc(pos, neg)
                if a is not None:
                    au.append(a)
                bd = sum((p - g) ** 2 for p, g in pr) / len(pr)
                br.append(bd)
                pv_ = sum(g for _, g in pr) / len(pr)
                cl = pv_ * (1 - pv_)
                if cl > 0:
                    bss.append(1 - bd / cl)
                e = ece_binary(pr)
                if e is not None:
                    ece.append(e)
        cell["mBAcc"] = round(sum(ba) / len(ba), 4) if ba else None
        cell["mAUC"] = round(sum(au) / len(au), 4) if au else None
        cell["mF1"] = round(sum(f1s) / len(f1s), 4) if f1s else None
        cell["mBrier"] = round(sum(br) / len(br), 4) if br else None
        cell["mBSS"] = round(sum(bss) / len(bss), 4) if bss else None
        cell["mECE"] = round(sum(ece) / len(ece), 4) if ece else None
        vv = []
        for r in rs:
            keys = choice_crit.get(r["question_id"], [])
            v = _cvec(r, keys)
            gi = _idx_of(r["gold"], keys) if v is not None else None
            if v is not None and gi is not None:
                vv.append((v, gi, bool(r.get("ok"))))
        cell.update({"cal_" + k: v for k, v in calib_vecs(vv, ordinal=False).items()})
    out["choice"] = cell

    qs = by.get("score", {})
    rs = [r for q in qs.values() for r in q]
    cell = {"n": len(rs)}
    if rs:
        def levels_idx(r):
            lv = score_levels.get(r["question_id"])
            p = r.get("probs")
            if not isinstance(p, dict) or not lv:
                return None
            out_ = {}
            for k, v in p.items():
                nk = norm(k)
                if nk in lv:
                    out_[lv.index(nk)] = out_.get(lv.index(nk), 0.0) + v
                else:
                    try:
                        i = int(nk)
                    except ValueError:
                        u = nk.upper()
                        if len(u) != 1 or not ("A" <= u <= "Z"):
                            return None
                        i = ord(u) - 64
                    if not (1 <= i <= len(lv)):
                        return None
                    out_[i - 1] = out_.get(i - 1, 0.0) + v
            return out_

        pred_e, pred_a, gold = [], [], []
        for r in rs:
            li = levels_idx(r)
            g = int(r["gold"])
            if li is None:
                if r.get("parsed") is not None:
                    pred_e.append(float(r["parsed"]))
                    pred_a.append(int(r["parsed"]))
                    gold.append(g)
                continue
            tot = sum(li.values())
            pred_e.append(sum(i * v for i, v in li.items()) / tot if tot else 0.0)
            pred_a.append(max(li, key=li.get))
            gold.append(g)
        n = len(pred_e)
        if n:
            mae = sum(abs(p - g) for p, g in zip(pred_e, gold)) / n
            gm = sum(gold) / n
            ssr = sum((p - g) ** 2 for p, g in zip(pred_e, gold))
            sst = sum((g - gm) ** 2 for g in gold)
            k = max(max(pred_a), max(gold)) + 1
            cell.update({"MAE": round(mae, 4), "R2": round(1 - ssr / sst, 4) if sst else None,
                         "QWK": round(qwk(pred_a, gold, k) * 100, 2) if qwk(pred_a, gold, k) is not None else None})
        else:
            cell.update({"MAE": None, "R2": None, "QWK": None})
        vv = []
        for r in rs:
            lv = score_levels.get(r["question_id"])
            li = levels_idx(r)
            if not lv or li is None or len(li) != len(lv):
                continue
            pv = [float(li.get(i, 0.0)) for i in range(len(lv))]
            t = sum(pv)
            if t <= 0:
                continue
            g = int(r["gold"])
            vv.append(([x / t for x in pv], g, bool(r.get("ok"))))
        cell.update({"cal_" + k: v for k, v in calib_vecs(vv, ordinal=True).items()})
    out["score"] = cell

    choice_qs = by.get("choice", {})
    noul_rs = [r for q in noul_by.values() for r in q]
    choice_rs = [r for q in choice_qs.values() for r in q]
    ovr = {}
    if noul_rs or choice_rs:
        both = noul_rs + choice_rs
        ovr["acc"] = round(sum(1 for r in both if r.get("ok")) / len(both), 4)
    pools = defaultdict(list)
    for qid, qr in choice_qs.items():
        for dis in choice_crit.get(qid, []):
            if dis in noul_dis:
                pools[dis] += [(qid, r) for r in qr]

    def bal(tp, npos, tn, nneg):
        rec = [x / y for x, y in ((tp, npos), (tn, nneg)) if y]
        return sum(rec) / len(rec) if rec else None

    ba = []
    for dis in sorted(noul_dis):
        br = []
        nr = [r for r in noul_by.get("finding_" + dis, []) if r.get("parsed") is not None]
        v = bal(sum(1 for r in nr if r["gold"] == 1 and r["parsed"] == 1),
                sum(1 for r in nr if r["gold"] == 1),
                sum(1 for r in nr if r["gold"] == 0 and r["parsed"] == 0),
                sum(1 for r in nr if r["gold"] == 0))
        if v is not None:
            br.append(v)
        tp = tn = npos = nneg = 0
        for qid, r in pools.get(dis, []):
            pd = _pred_dis(r, choice_crit, letter_of)
            if dis_of[qid](r) == dis:
                npos += 1
                tp += 1 if (r.get("parsed") is not None and pd == dis) else 0
            else:
                nneg += 1
                tn += 1 if (r.get("parsed") is not None and pd != dis) else 0
        v = bal(tp, npos, tn, nneg)
        if v is not None:
            br.append(v)
        if br:
            ba.append(sum(br) / len(br))
    if ba:
        ovr["mBAcc"] = round(sum(ba) / len(ba), 4)
    ovp = []
    for r in noul_rs + choice_rs:
        p = r.get("probs")
        if isinstance(p, dict) and p:
            ovp.append((max(p.values()), bool(r.get("ok"))))
    if ovp:
        ovr["AURC"] = round(aurc(ovp), 4)
        ovr["topECE"] = round(ece_top(ovp), 4)
        ovr["n_cal"] = len(ovp)
    out["overall"] = ovr
    return out


def main():
    argv = list(sys.argv[1:])
    if "--cmp" in argv:
        return cmp_main(argv)
    noul_dir = None
    if "--noul-dir" in argv:
        i = argv.index("--noul-dir")
        noul_dir = argv[i + 1]
        del argv[i:i + 2]
    inp = argv[1] if len(argv) > 1 else str(
        Path(__file__).resolve().parent.parent / "out_dataset_final/usable_test.jsonl")
    print(json.dumps(compute(argv[0], inp, noul_dir), ensure_ascii=False, indent=1))
    return 0


CMP_COLS = [("noul", ["acc", "mBrier", "mBSS", "mECE", "cal_Brier", "cal_BSS",
                       "cal_topECE", "cal_AURC", "cal_n_cal"]),
            ("choice", ["acc", "mBrier", "mBSS", "mECE", "cal_Brier", "cal_BSS",
                        "cal_topECE", "cal_AURC", "cal_n_cal"]),
            ("score", ["MAE", "QWK", "cal_Brier", "cal_BSS", "cal_RPS", "cal_RPS_clim",
                       "cal_RPSS", "cal_topECE", "cal_AURC", "cal_n_cal"]),
            ("overall", ["acc", "AURC", "topECE", "n_cal"])]


def cmp_main(argv):
    """python3 analysis/verdict_metrics.py --cmp <dir>[:标签] ... [--input <jsonl>]

    同一份输入表、同一套口径，把几个 run 的校准块并排打出来（RL 臂 vs 它的基座 vs
    racd1）。`n_cal` 是进了校准块的行数 —— 各臂分母必须一样，不一样就别比。
    """
    i = argv.index("--cmp")
    toks = argv[i + 1:]
    inp = str(Path(__file__).resolve().parent.parent / "out_dataset_final/usable_test.jsonl")
    if "--input" in toks:
        j = toks.index("--input")
        inp = toks[j + 1]
        del toks[j:j + 2]
    runs = [(t.split(":", 1)[1] if ":" in t else Path(t.split(":", 1)[0]).parent.name,
             t.split(":", 1)[0]) for t in toks]
    ops = {}
    for lb, d in runs:
        op = 0
        for mj in sorted(Path(d).glob("metrics_*.json")):
            try:
                op = int(json.load(open(mj, encoding="utf-8")).get("option_perm", 0) or 0)
            except (OSError, ValueError):
                continue
            break
        ops[d] = op
    runs = [(f'{lb}[π{ops[d]}]' if ops[d] else lb, d) for lb, d in runs]
    if len(set(ops.values())) > 1:
        print('⚠ 这张表混了选项轮转口径（π0 = 未轮转 / πN = 按 sha256(N, qid) 轮转）：'
              '只有**同一个 ckpt** 的不同 π 之间可比（置换不变性闸门）；'
              'πN≠0 的行不许与已报数的 π0 臂比大小。', flush=True)
    res = [(lb, compute(d, inp)) for lb, d in runs]
    for kind, cols in CMP_COLS:
        print(f'\n===== {kind} =====')
        print('run'.ljust(24) + ''.join(f'{c:>14}' for c in cols))
        for lb, r in res:
            cell = r.get(kind) or {}
            print(lb[:23].ljust(24) + ''.join(
                f'{("--" if cell.get(c) is None else cell.get(c)):>14}' for c in cols))
    print('\n注：cal_Brier/cal_BSS 是**多类**（本支候选集上）按 K 分桶加权；mBrier/mBSS 是'
          '逐病 OvR（冻结口径，与 mAUC 同行集）。mECE=逐病二分类可靠性误差宏平均，'
          'cal_topECE=支级 top-label（p_max，10 等宽箱）。RPS 只在 score 出（choice 无序）。'
          'AURC 必须与同支 acc 连着读：AURC−acc 才是拒答净增益。')
    return 0


def _pred_dis(r, choice_crit, letter_of):
    """行的 parsed 对应的疾病名（parsed 可能是疾病名或字母）。"""
    keys = choice_crit.get(r["question_id"], [])
    p = r.get("parsed")
    if p is None:
        return None
    if p in keys:
        return p
    lm = {v: k for k, v in letter_of[r["question_id"]].items()}
    return lm.get(p)


if __name__ == "__main__":
    main()
