"""三级地板里的 ①②（plan §3 表、步骤 5–6「都不需要训练」）。

① `majority` —— state-only 的**无训练退化形**。逐题在**拟合集**（现在默认是合并后的
   `train+val`，见 `loop --train-split`）上数类频，评估集照搬多数类。它就是
   「population 捷径地板」在零参数下的最强形态：③ 级模型若连它都赢不了，
   说明没学到任何东西。**注意它的 acc 就是「题面先验」的上限，而它的逐题
   bal_acc / AUROC 恒等于 0.5**（组内常数预测），所以判 ③ 有没有用上影像要看
   bal_acc / 逐题 AUROC，不是看 acc。（③ 的 state-only 模式另有 `dataset.n_vis=0`
   走同一份训练代码，那是**训练过的** ①，与这里这个零参数版并存、分别报告。）
② `radar-cosine` —— RADAR 自带分类读出（`logits = image_feat @ text_feat.T`，逐病二元概率），
   **无 LLM**。读 stage 1 缓存的 `probs`/`prob_keys`（官方 20 项逐例概率，plan §1）。
   覆盖范围只有「noul 题 ∧ 病名 ∈ 官方 20 项」—— 其余题型/9 个 RADAR 没有的病
   一律计 uncovered 并逐类计数，**不猜、不填零**。

两者的产物形状一致：`per_question` 汇总 + `pred` 逐例（`{question: {index: key}}`，
② 另带 `p_pos` = 该题 Yes 列的概率，索引恒为 1；符号版 a/b 同列序），供 `metrics.py`
做与 ③ 的配对显著性（McNemar）。

落盘位置由调用方给（`out_train/…`）—— **不写 `out_v3/`**：出厂目录只读，
plan §6-8 那句「写进 manifest 的 floors」待用户点头后才做。
"""
from __future__ import annotations

import collections
import json

from . import pack

PROB_KEYS_20 = None


def _target_key(ds, i):
    ex = ds.examples[i]
    rec = ds._record(ex)
    ins = rec['questions'][ex[2]]
    space, _ = ds._space(ex[2], ins)
    ai = pack.target_index(space, rec['targets'][ex[2]])
    return space, space.keys[ai]


def _iter_targets(ds):
    """不建序列，只出 (study_id, question, space, target_key)——440k 次实测秒级。

    键是 `study_id`（与 `loop._predict`、`metrics.collect` 同一约定）；
    用数据集下标当键会在换 split/换过滤条件时静默错位。
    """
    for i in range(len(ds)):
        ex = ds.examples[i]
        rec = ds._record(ex)
        ins = rec['questions'][ex[2]]
        space, _ = ds._space(ex[2], ins)
        ai = pack.target_index(space, rec['targets'][ex[2]])
        yield ex[1], ex[2], space, space.keys[ai]


def majority_floor(train_ds, eval_ds):
    """① 逐题多数类（train 上数，eval 上应用）。ties 按 key 排序取首个 —— 确定性。"""
    cnt = collections.Counter()
    for _, q, _, key in _iter_targets(train_ds):
        cnt[(q, key)] += 1
    maj = {}
    for (q, key), n in cnt.items():
        if q not in maj or (n, key) > (maj[q][1], maj[q][0]):
            maj[q] = (key, n)

    per_q = collections.defaultdict(lambda: {'n': 0, 'n_covered': 0, 'n_correct': 0,
                                             'dist': collections.Counter()})
    pred = collections.defaultdict(dict)
    for sid, q, _, tkey in _iter_targets(eval_ds):
        r = per_q[q]
        r['n'] += 1
        r['dist'][tkey] += 1
        if q in maj:
            pk = maj[q][0]
            r['n_covered'] += 1
            r['n_correct'] += int(pk == tkey)
            pred[q][str(sid)] = pk
    out = {}
    for q, r in per_q.items():
        out[q] = {'n': r['n'], 'n_covered': r['n_covered'], 'n_correct': r['n_correct'],
                  'acc': (r['n_correct'] / r['n_covered']) if r['n_covered'] else None,
                  'majority': maj.get(q, (None, 0))[0],
                  'base_rate': (maj.get(q, (None, 0))[1] / sum(
                      v for (qq, _), v in cnt.items() if qq == q)) if q in maj else None,
                  'dist': dict(sorted(r['dist'].items()))}
    n_tot = sum(r['n'] for r in out.values())
    n_cov = sum(r['n_covered'] for r in out.values())
    return {'name': 'majority', 'source': f'train counts ({train_ds.split})',
            'n_eval': n_tot, 'n_covered': n_cov,
            'acc_micro_covered': (sum(r['n_correct'] for r in out.values()) / n_cov) if n_cov else None,
            'per_question': out, 'pred': {q: dict(d) for q, d in pred.items()}}


def _prob_lookup(ds):
    """sid -> (prob_keys, probs) 的带缓存读取（npz 逐例开太贵）。"""
    cache = {}

    def get(sid):
        if sid not in cache:
            cache[sid] = ds.npz_probs(sid)
        return cache[sid]
    return get


def radar_floor(eval_ds):
    """② RADAR 官方逐病二元概率。只有 noul 题 ∧ 病名在 `prob_keys` 里才给预测。"""
    if eval_ds.feats_mode != 'cache':
        extra = '（纯 2D 臂不读 3D 特征）' if eval_ds.feats_mode == 'none' else ''
        raise SystemExit(f'radar-cosine 地板需要 feats=cache 的 npz probs，'
                         f'当前 feats={eval_ds.feats_mode}{extra}')
    get = _prob_lookup(eval_ds)
    per_q = collections.defaultdict(lambda: {'n': 0, 'n_covered': 0, 'n_correct': 0,
                                             'dist': collections.Counter()})
    pred = collections.defaultdict(dict)
    ppos = collections.defaultdict(dict)
    why = collections.Counter()
    for i in range(len(eval_ds)):
        ex = eval_ds.examples[i]
        sid = ex[1]
        rec = eval_ds._record(ex)
        ins = rec['questions'][ex[2]]
        q = ex[2]
        r = per_q[q]
        r['n'] += 1
        space, _ = eval_ds._space(q, ins)
        ai = pack.target_index(space, rec['targets'][ex[2]])
        tkey = space.keys[ai]
        r['dist'][tkey] += 1
        if space.kind != 'noul':
            why['not_noul'] += 1
            continue
        dis = ins.get('disease')
        if not dis:
            why['no_disease_field'] += 1
            continue
        keys, probs = get(ex[1])
        if dis not in keys:
            why['disease_not_in_radar_20'] += 1
            continue
        p = float(probs[keys.index(dis)])
        pk = space.keys[1] if p >= 0.5 else space.keys[0]
        r['n_covered'] += 1
        r['n_correct'] += int(pk == tkey)
        pred[q][str(sid)] = pk
        ppos[q][str(sid)] = p
    out = {}
    for q, r in per_q.items():
        out[q] = {'n': r['n'], 'n_covered': r['n_covered'], 'n_correct': r['n_correct'],
                  'acc': (r['n_correct'] / r['n_covered']) if r['n_covered'] else None,
                  'dist': dict(sorted(r['dist'].items()))}
    n_tot = sum(r['n'] for r in out.values())
    n_cov = sum(r['n_covered'] for r in out.values())
    return {'name': 'radar-cosine', 'source': 'stage 1 npz probs (official 20-item readout)',
            'n_eval': n_tot, 'n_covered': n_cov,
            'acc_micro_covered': (sum(r['n_correct'] for r in out.values()) / n_cov) if n_cov else None,
            'uncovered_why': dict(why),
            'per_question': out,
            'pred': {q: dict(d) for q, d in pred.items()},
            'p_pos': {q: dict(d) for q, d in ppos.items()}}


def write(path, *floors):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({fl['name']: fl for fl in floors}, f, ensure_ascii=False, indent=1)
    return path
