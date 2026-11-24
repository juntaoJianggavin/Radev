"""逐题指标 + 与 ①② 地板的配对显著性（plan §6 第 8 条：③ 未显著超过两地板则拒绝报数）。

指标按题型（`pack.AnswerSpace.kind`）：

* `noul` / `choice` —— **acc**（报告口径；逐题 macro 另算）；
* `score` —— **level MAE** / **QWK**（有序档位；不再用 exact-match acc 报分）。
  MAE 有候选概率时先用 `Σ_k p_k · k` 算期望档位（可为小数），再算 `|期望档位 − gold|`；
  没有概率才退回 argmax 的整数档位差。QWK 仍按 argmax 的硬标签。
  bal_acc / AUROC / 混合 total 只留在 metrics JSON 作内部诊断，不进报告。

显著性：与每个地板的**配对子集**（两侧都给了预测的例）上用 McNemar 精确/近似检验。
b = 模型对而地板错，c = 模型错而地板对；判定固定三值：`③显著超过` / `③显著低于` /
`未显著超过`（α=0.05）。**只要存在显著低于就必须在报告里写明**。

`render_md` 除逐题/逐族外还切一列**按 rule_class**（`collect` 从 `examples[i][4]` 带出标签的
规则名）—— 回收来的暗示阴性挂 `implied_negative_v2`，它和三个老口径放在一起对比，
用来盯「学的是报告写法还是病灶」这件事。
"""
from __future__ import annotations

import collections
import math

from . import pack

ALPHA = 0.05
_PREFICES = ('finding_', 'score_', 'side_', 'organ_', 'acq_', 'prog_', 'triage_')


def _mcnemar(b, c):
    """配对二项检验的双侧 p（b、c = 两个方向的不一致对数）。"""
    n = b + c
    if n == 0:
        return 1.0
    if n <= 100:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)
    chi2 = (abs(b - c) - 1.0) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2.0))


def _auroc(pos, neg):
    """秩和法 AUC；ties 取平均秩。pos/neg 为分数列表。"""
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


def _qwk(cm, k):
    n = sum(sum(r) for r in cm)
    if n == 0 or k < 2:
        return None
    w = [[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)]
    ht = [sum(cm[i]) for i in range(k)]
    hp = [sum(cm[i][j] for i in range(k)) for j in range(k)]
    num = sum(w[i][j] * cm[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * ht[i] * hp[j] / n for i in range(k) for j in range(k))
    return None if den == 0 else 1.0 - num / den


def fam_of(q):
    for pre in _PREFICES:
        if q.startswith(pre):
            return pre.rstrip('_')
    return q


def collect(ds, pred, p_pos=None, p_dist=None):
    """eval 数据集 + 预测 `{question: {study_id: key}}`（+ 可选 p_pos / p_dist）-> 行表。

    行 = (study_id, question, prefix, kind, y_key, y_idx, p_key, p_idx, ordinal, p_pos,
          rule_class, p_dist)。
    `p_dist` 是 score 题的候选概率向量（按 answer space 的档位顺序），有它时先把分布
    压成期望档位 `Σ p_k·k`（可为小数），再与 gold 作差；没有时退回 argmax 的档位差。
    预测缺该例 ⇒ `p_key=None`（未覆盖：进 n、不进 acc 分母）。
    键一律 study_id（见 loop._predict / floor.*），与 `compare()` 的地板预测同一约定。
    `rule_class` 直接取 `ds.examples[i][4]`（名单那一列，不是这里现推的）：只有 `finding_*`
    非空，2026-10-12 起它有 6 个取值 —— 五个极性规则 + `implied_negative_v2`（逐绑定回收的
    暗示阴性）。切这一列是用户的明确要求：这类标签的含义是「报告没提 ⇒ 没有」，
    要能看见模型在**这一列**上与其他规则差多少，而不是被 finding 族的总数盖住。
    """
    rows = []
    for i in range(len(ds)):
        ex = ds.examples[i]
        rec = ds._record(ex)
        ins = rec['questions'][ex[2]]
        space, _ = ds._space(ex[2], ins)
        ai = pack.target_index(space, rec['targets'][ex[2]])
        p_key = pred.get(ex[2], {}).get(str(ex[1]))
        p_idx = space.keys.index(p_key) if p_key is not None else None
        pp = None
        if p_pos is not None:
            pp = p_pos.get(ex[2], {}).get(str(ex[1]))
        pd = None
        if p_dist is not None:
            pd = p_dist.get(ex[2], {}).get(str(ex[1]))
        rows.append((ex[1], ex[2], ex[3], space.kind, space.keys[ai], ai,
                     p_key, p_idx, space.ordinal, pp, ex[4], pd))
    return rows


def summarize(rows, kind):
    """一个（问题, kind）行组 -> 指标 dict。"""
    judged = [r for r in rows if r[6] is not None]
    out = {'n': len(rows), 'n_covered': len(judged), 'acc': None, 'bal_acc': None,
           'auroc': None, 'qwk': None, 'mae': None, 'mae_hard': None,
           'r2': None, 'r2_hard': None, 'auroc_scored': False}
    if not judged:
        return out
    nc = len(judged)
    out['acc'] = sum(int(r[4] == r[6]) for r in judged) / nc
    cls = sorted({r[4] for r in judged})
    rec = [sum(int(r[4] == c and r[6] == c) for r in judged) /
           max(1, sum(int(r[4] == c) for r in judged)) for c in cls]
    out['bal_acc'] = sum(rec) / len(rec) if rec else None
    if kind == 'noul':
        have_p = [r for r in judged if r[9] is not None]
        if have_p:
            out['auroc_scored'] = True
            out['auroc'] = _auroc([float(r[9]) for r in have_p if r[5] == 1],
                                  [float(r[9]) for r in have_p if r[5] == 0])
        else:
            out['auroc'] = _auroc([1.0 if r[7] == 1 else 0.0 for r in judged if r[5] == 1],
                                  [1.0 if r[7] == 1 else 0.0 for r in judged if r[5] == 0])
        out['n_pos'] = sum(int(r[5] == 1) for r in judged)
        out['n_neg'] = sum(int(r[5] == 0) for r in judged)
    if kind == 'score':
        k = max([r[5] for r in judged] + [r[7] for r in judged]) + 1
        cm = [[0] * k for _ in range(k)]
        for r in judged:
            cm[r[5]][r[7]] += 1
        out['qwk'] = _qwk(cm, k)
        preds = []
        for r in judged:
            pd = r[11] if len(r) > 11 else None
            p = None
            if pd:
                z = sum(float(x) for x in pd)
                if z > 0:
                    p = sum(float(x) * i for i, x in enumerate(pd)) / z
            preds.append(p)
        err = 0.0
        for p, r in zip(preds, judged):
            err += abs((p if p is not None else float(r[7])) - r[5])
        out['mae'] = err / nc
        out['mae_hard'] = sum(abs(r[5] - r[7]) for r in judged) / nc
        out['n_prob'] = sum(p is not None for p in preds)
        ys = [r[5] for r in judged]
        mu = sum(ys) / nc
        sst = sum((y - mu) ** 2 for y in ys)
        sse = sum(((p if p is not None else float(r[7])) - r[5]) ** 2
                  for p, r in zip(preds, judged))
        out['r2'] = 1.0 - sse / sst if sst > 0 else None
        out['r2_hard'] = (1.0 - sum((r[7] - r[5]) ** 2 for r in judged) / sst
                          if sst > 0 else None)
    return out


def evaluate(rows):
    """行表 -> (per_question, by_family, by_rule, total, by_kind)。

    `by_kind` 是报告口径：noul/choice 用 acc，score 用 level MAE / QWK。`total`
    仍是跨题型的混合值，只留给旧 JSON 消费方，报告不再使用。

    per_q 键 = (question, kind)；by_rule 键 = (rule_class, kind)，**只有 `finding_*` 有这一列**
    （其余前缀的 rule_class 是空串，它们已经由 by_family 覆盖，不在这里重复一份总体）。

    🔴 rule_class 是**条目**（study, question）的属性，不是题名的属性：同一个题名在别的卷里
    可以由别的极性规则凭据成立 —— `finding_renal_hypodensities` 在一卷里来自明确否定句、
    在另一卷里来自断言句，完全正常。所以这一列**直接按行切**，不经过 per_q 的分组
    （旧代码在这里 assert「一个题名只属一个 rule_class」，2026-10-14 的 B 支自检里被真实
    数据否证：三条 rule_class 同时出现在同一题名上。那个假设错在把名单的一列当成题目的属性）。
    """
    per_q = collections.OrderedDict()
    for r in rows:
        per_q.setdefault((r[1], r[3]), []).append(r)
    fam = collections.OrderedDict()
    rule = collections.OrderedDict()
    for (q, kind), rs in per_q.items():
        fam.setdefault((fam_of(q), kind), []).extend(rs)
    for r in rows:
        if r[10]:
            rule.setdefault((r[10], r[3]), []).append(r)
    per_q = collections.OrderedDict((k, summarize(v, k[1])) for k, v in per_q.items())
    fam = collections.OrderedDict((k, summarize(v, k[1])) for k, v in fam.items())
    rule = collections.OrderedDict((k, summarize(v, k[1])) for k, v in rule.items())
    total = summarize(rows, None)
    by_kind = collections.OrderedDict()
    for kind in ('noul', 'choice', 'score'):
        rs = [r for r in rows if r[3] == kind]
        if rs:
            by_kind[kind] = summarize(rs, kind)
    return per_q, fam, rule, total, by_kind


def compare(rows, floor_pred):
    """模型行表 vs 地板预测 -> 逐问题 McNemar（配对子集 = 两侧都给预测的例）。"""
    out = collections.OrderedDict()
    for r in rows:
        if r[3] == 'score':
            continue
        q = r[1]
        pk = floor_pred.get(q, {}).get(str(r[0]))
        if pk is None or r[6] is None:
            continue
        yk = r[4]
        d = out.setdefault(q, {'b': 0, 'c': 0, 'n_paired': 0})
        d['n_paired'] += 1
        d['b'] += int(r[6] == yk and pk != yk)
        d['c'] += int(r[6] != yk and pk == yk)
    for q, d in out.items():
        p = _mcnemar(d['b'], d['c'])
        d['p'] = p
        d['verdict'] = ('③显著超过' if p < ALPHA and d['b'] > d['c'] else
                        '③显著低于' if p < ALPHA and d['c'] > d['b'] else '未显著超过')
    return out


def _f(x):
    return '—' if x is None else f'{x:.4f}'


def _kind_acc(vs):
    """按覆盖数加权的 acc（noul/choice 才用）。"""
    num = sum(v['acc'] * v['n_covered'] for v in vs if v['acc'] is not None)
    den = sum(v['n_covered'] for v in vs if v['acc'] is not None)
    return num / den if den else None


def _macro_cell(vs):
    """逐题 macro：全部 question_id 等权 / 只取 n≥100 的题。"""
    accs = [v['acc'] for v in vs if v['acc'] is not None]
    big = [v['acc'] for v in vs if v['acc'] is not None and v['n'] >= 100]
    if not accs:
        return '—'
    if not big:
        return f'{sum(accs) / len(accs):.4f} / —'
    return f'{sum(accs) / len(accs):.4f} / {sum(big) / len(big):.4f}'


def render_md(title, per_q, by_family, by_rule, by_kind, floor_cmps=None, notes=()):
    """报告口径：是非/选择题报 acc，score 报 level MAE / QWK；不报跨题型混合 acc。"""
    L = [f'# {title}', '']
    for n in notes:
        L.append(f'> {n}')
    if notes:
        L.append('')
    L.append('> 指标口径：是非题（noul）与选择题（choice）报 acc；score 题报 level MAE / QWK。'
             '不报跨题型的混合 acc。`MAE*` 有候选概率时先用 Σ p_k·k 算期望档位（可为小数），'
             '再算 |期望档位 − gold|；没有概率的例才退回 argmax 档位差。QWK 仍按 argmax 硬标签。')
    L.append('')
    L.append('| 题型 | n | 覆盖 | acc | 逐题 macro（全部 / n≥100） | MAE* | QWK |')
    L.append('|---|---|---|---|---|---|---|')
    for kind in ('noul', 'choice'):
        vs = [v for (q, k), v in per_q.items() if k == kind]
        if not vs:
            continue
        L.append('| {} | {} | {} | {} | {} | — | — |'.format(
            kind, sum(v['n'] for v in vs), sum(v['n_covered'] for v in vs),
            _f(_kind_acc(vs)), _macro_cell(vs)))
    s = by_kind.get('score')
    if s:
        L.append('| score | {} | {} | — | — | {} | {} |'.format(
            s['n'], s['n_covered'], _f(s['mae']), _f(s['qwk'])))
        if s.get('n_prob') is not None:
            L.append('')
            L.append(f'> score MAE 期望档位覆盖 {s["n_prob"]}/{s["n_covered"]}。')
    L.append('')
    L.append('| 范围 | 题型 | n | 覆盖 | acc | MAE* | QWK |')
    L.append('|---|---|---|---|---|---|---|')
    rows = [(f'{k[0]}/{k[1]}', k[1], v) for k, v in by_family.items()]
    rows += [(f'{k[0]}/{k[1]}', k[1], v) for k, v in per_q.items()]
    for name, kind, s in rows:
        if kind == 'score':
            L.append('| {} | {} | {} | {} | — | {} | {} |'.format(
                name, kind, s['n'], s['n_covered'], _f(s['mae']), _f(s['qwk'])))
        else:
            L.append('| {} | {} | {} | {} | {} | — | — |'.format(
                name, kind, s['n'], s['n_covered'], _f(s['acc'])))
    L.append('')
    auc_rows = [((q, kind), s) for (q, kind), s in per_q.items()
                if kind == 'noul' and q.startswith('finding_')
                and s.get('auroc') is not None and s.get('auroc_scored')]
    if auc_rows:
        auc_rows.sort(key=lambda x: x[1]['auroc'])
        L.append('## 逐疾病 AUC（finding_*，noul；每个疾病只用自己的正负例）')
        L.append('')
        L.append('| 疾病 | n | pos | neg | AUC | acc |')
        L.append('|---|---|---|---|---|---|')
        for (q, _), s in auc_rows:
            L.append('| {} | {} | {} | {} | {} | {} |'.format(
                q, s['n'], s.get('n_pos', '—'), s.get('n_neg', '—'),
                _f(s['auroc']), _f(s['acc'])))
        macro = sum(s['auroc'] for _, s in auc_rows) / len(auc_rows)
        L.append('')
        L.append(f'宏平均（{len(auc_rows)} 个疾病）：AUC={macro:.4f}')
        L.append('')
        L.append('> AUC 只在该疾病自己的正负例上算，不做跨疾病合并；少数类 <10 的疾病 AUC 不稳，'
                 '读的时候看 pos/neg。它与 acc 的阈值口径不同。')
        L.append('')
    L.append('## 按 rule_class（finding 一族按标签的可信度分层）')
    L.append('')
    L.append('| rule_class | n | 覆盖 | acc |')
    L.append('|---|---|---|---|')
    for (rc, kind), s in by_rule.items():
        L.append('| {} | {} | {} | {} |'.format(
            rc, s['n'], s['n_covered'], _f(s['acc'] if kind != 'score' else None)))
    L.append('')
    L.append('`implied_negative_v2` = 暗示阴性里**过两道闸**的那批（所属绑定的两读者齐声反对率'
             ' ≤ 阈值，且逐条过当前判据档位；名单 `out_inventory/implied_negative_v2_items_keep.tsv`）。'
             '它的标签含义是「报告没提这个病 ⇒ 没有」，所以模型可以在**读报告写法**上拿分而不看病灶：'
             '这一列明显高于同一次运行里 `asserted_positive` / `explicit_negative` 那种'
             '「文本明说」的档，**不能**读成病灶判别能力，只能读成它对报告措辞的拟合。'
             '`implied_negative`（未回收）永远不出现 —— 它在名单里全是 keep=0，进不了训练也进不了评估。'
             '想比较「收与不收」要跑两次（`--rules` 带/不带 `implied_negative_v2`），'
             '一次运行里没有对照组。')
    L.append('')
    for fname, cmp in (floor_cmps or {}).items():
        if fname == 'majority':
            continue
        L.append(f'## vs 地板 `{fname}`（McNemar，α={ALPHA}）')
        L.append('')
        L.append('> McNemar 只用于是非/选择题的 exact match；score 走 MAE/QWK，不进此表。')
        L.append('')
        L.append('| 问题 | 配对例数 | b(③对/地板错) | c(③错/地板对) | p | 判定 |')
        L.append('|---|---|---|---|---|---|')
        over = under = 0
        for q, d in cmp.items():
            over += d['verdict'] == '③显著超过'
            under += d['verdict'] == '③显著低于'
            L.append(f'| {q} | {d["n_paired"]} | {d["b"]} | {d["c"]} | {d["p"]:.3g} | '
                     f'{d["verdict"]} |')
        n_q = len(cmp)
        L.append('')
        L.append(f'汇总：显著超过 {over}/{n_q}，显著低于 {under}/{n_q}，其余未显著超过。')
        L.append('')
    return '\n'.join(L) + '\n'
