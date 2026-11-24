#!/usr/bin/env python
"""Part A 步骤 4 —— stage 1 缓存：RADAR 冻结编码器的逐 study 产物。

    plan（CPU，秒级）                                      run（GPU，逐片 worker）
      out_dataset_final_sym/{train_val,test}.json ──┐       shards/part_<k>.txt
      keep TSV（split 列）+ geometry_scan.json ─────┼─> shards/ └─> <sha12>/<split>/<sid>.npz
                                                     │   part_<k>.txt
                                                     └   plan.json / .ngpu

产物布局（根目录默认 feature_cache/，与 parity_demo / geometry_scan 同根）：
    <sha12>/meta.json             encoder 身份 + 特征列语义（a15 的锚点）
    <sha12>/<split>/<sid>.npz     逐 study 张量，见 _save_npz
    shards/part_<k>.txt           每行 `split/sid`（LPT 按 n_windows 降序贪心均衡）
    shards/.ngpu                  切分时的片数（片数-切分绑定；与 run_review_8gpu.sh 同款）
    shards/plan.json              计划元数据（逐片 / 逐 split 计数 + 源文件 sha256）
    logs/summary_shard<k>.json    每片汇总（脚本侧把 stdout 写 logs/shard_<k>.log）

npz 键（列的定序语义靠 meta.json 的 organs / prob_items，不靠任何散文）：
    feats      float16[36,256]  按 RE.ORGANS 定序的 L2-normalized 器官向量；缺器官整行 0
    has_feat   bool[36]         该器官这卷有向量 —— 缺器官不能与「恰为全 0 的向量」混淆
    sizes      int32[36]        器官体素计数（organ_sizes）；没这个器官 = 0
    whole_feat float16[256]     整图分支向量（L2-normalized）；has_whole=False 时全 0
    has_whole  bool              整图向量是否有效（上游整图支路无滑窗、每卷恰好评一次）
    probs      float32[K]       官方 20 征象的正例概率（K ≤ 20，缺项不进数组）
    prob_keys  str[K]           probs 第 i 项是哪个征象 —— 对齐 probs 的唯一凭据
    sag_voxels float32          stitched_mask 胆囊体素数（上游 :548-561 那一列）
    n_windows  int32            实际滑窗数（几何扫描给的是上界，crop 之后才是真值）
    organ_box  int16[36,6]      器官分位点 bbox（z0,z1,y0,y1,x0,x1 含界；-1=mask 里无该类）
    organ_zprof int32[36,D]     逐层器官体素计数（D=96 padded 层数；mask 感知选层的唯一依据）

三条不许含糊的点：
1. a15：meta.json 的 (encoder_identity, ckpt_sha256, n_tensors_visual, organs, prob_items,
   text_emb_sha256) 与本次加载的不一致 → 拒绝续跑。特征列的语义错了不会报错，
   只会让下游学一列错位的东西。
2. 原子写：npz 与 meta.json 都走 tmp + os.replace —— 中断留下半个文件，下次续跑就会
   拿它当真值（run_review 的 _check_resume_compat 同一课）。
3. 逐例失败不致命（记 failed_shard<k>.txt 后继续），但会在片尾汇总与脚本对账里显形。

`--encoder radar_ft` 的两处后果（本次决定「只跑 finetuned」，a17）：
* 中文文本嵌入 `.pt` 与英文文本塔不同源 ⇒ probs 无法在本步产出。这里**不加载文本塔**、
  以 `test_items=[]` 跑滑窗，probs 写空数组，meta 记 `probs_pending` + 待重算的 20 项。
  概率是 (feats, text_feat, temp) 的纯函数 ⇒ ② 级地板从缓存离线重算（任务 #60），
  不需要第二遍 GPU。
* 整图向量 `whole_feat` 只能在这一步产出（预处理与器官支路不同，见 volume.py）。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from radev import config
from radev.train import radar_encoder as RE
from radev.train import visual_frontend as VF

DEFAULT_OUT = os.path.join(_REPO, 'feature_cache')
DEFAULT_TRAIN_JSON = os.path.join(_REPO, 'data', 'iid', 'train_val.json')
DEFAULT_EVAL_JSON = os.path.join(_REPO, 'data', 'iid', 'test.json')
KEEP_TSVS = (
    os.path.join(_REPO, 'question_keep.tsv'),
    os.path.join(_REPO, 'newq_keep.tsv'),
)
DEFAULT_SCAN = os.path.join(_REPO, 'geometry_scan.json')
_CKPT_ROOT = os.environ.get('RADAR_CKPT_ROOT') or os.path.join(_REPO, 'radar_ckpt')
CKPT_BY_IDENTITY = {
    'pretrain': os.path.join(_CKPT_ROOT, 'checkpoint_radar_pretrain.pth'),
    'radar_ft': os.path.join(_CKPT_ROOT, 'checkpoint_radar_plus_finetuned.pth'),
}
DEFAULT_ENCODER = 'radar_ft'
DEFAULT_TEXT_EMB = os.path.join(_CKPT_ROOT, 'infer_text_embedding.pt')


def _append(path: str, line: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _sha256(path: str) -> str:
    return RE.sha256_of(path)



def _iter_json_rows(path: str) -> list[dict]:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise SystemExit(f'{path} 不是 JSON 数组（终版导出链的产物是数组）')
    return raw


def _row_study(rec: dict, path: str, i: int) -> tuple[str, str]:
    rid = rec.get('id') or ''
    if '::' not in rid:
        raise SystemExit(f'{path} 第 {i} 条 id 不是 study::question：{rid!r}')
    sid, q = rid.split('::', 1)
    want = {f'iid/{sid}.nii.gz', f'ood/{sid}.nii.gz'}
    if rec.get('image') not in want:
        raise SystemExit(f'{path} 第 {i} 条 image={rec.get("image")!r} 不在 {sorted(want)}'
                         '（release 相对图像路径口径）')
    return sid, q


def _final_study_split(train_json: str, eval_json: str) -> tuple[dict[str, str], int, int]:
    """终版符号 JSON -> {study: split}；split 由 keep 名单的 split 列定（与训练侧同源）。"""
    from radev.train.dataset import _load_export_annotations

    train_rows = _iter_json_rows(train_json)
    if not train_rows:
        raise SystemExit(f'{train_json} 是空数组')
    keys: list[tuple[str, str]] = []
    seen_keys: set[tuple[str, str]] = set()
    for i, rec in enumerate(train_rows):
        if rec.get('answer_format') != 'symbol':
            raise SystemExit(f'{train_json} 第 {i} 条不是符号版（answer_format='
                             f'{rec.get("answer_format")!r}）—— plan 必须对训练数据面')
        key = _row_study(rec, train_json, i)
        if key in seen_keys:
            raise SystemExit(f'{train_json} 第 {i} 条 (study, question) 重复：{key}')
        seen_keys.add(key)
        keys.append(key)

    ann = _load_export_annotations(seen_keys)
    study_split: dict[str, str] = {}
    for sid, q in keys:
        a = ann.get((sid, q))
        sp = a[2] if a else ''
        if sp not in config.SPLITS:
            raise SystemExit(f'{train_json} 的 {(sid, q)} 不在 keep 名单的 split 列里 —— '
                             '导出链与 question_keep.tsv 不同源')
        old = study_split.setdefault(sid, sp)
        if old != sp:
            raise SystemExit(f'study {sid} 同时属于 {old}/{sp} —— split 不是逐卷唯一的')

    test_rows = _iter_json_rows(eval_json)
    if not test_rows:
        raise SystemExit(f'{eval_json} 是空数组')
    seen_test: set[tuple[str, str]] = set()
    for i, rec in enumerate(test_rows):
        if rec.get('answer_format') != 'symbol':
            raise SystemExit(f'{eval_json} 第 {i} 条不是符号版（answer_format='
                             f'{rec.get("answer_format")!r}）')
        key = _row_study(rec, eval_json, i)
        if key in seen_test:
            raise SystemExit(f'{eval_json} 第 {i} 条 (study, question) 重复：{key}')
        seen_test.add(key)
        sid, _ = key
        old = study_split.setdefault(sid, 'test')
        if old != 'test':
            raise SystemExit(f'study {sid} 同时出现在 {old} 与 test')
    return study_split, len(train_rows), len(test_rows)


def _collect(scan_path: str, train_json: str = DEFAULT_TRAIN_JSON,
             eval_json: str = DEFAULT_EVAL_JSON
             ) -> tuple[list[tuple[str, str, int]], dict[str, int], int, tuple[int, int]]:
    """从终版符号 JSON + keep split + geometry_scan 现算全量条目；对不上即 SystemExit。"""
    with open(scan_path, encoding='utf-8') as f:
        scan = json.load(f)
    idx = {r['study']: r for r in scan['rows']}
    study_split, n_train, n_test = _final_study_split(train_json, eval_json)

    entries: list[tuple[str, str, int]] = []
    by_split: dict[str, int] = {}
    for sid in sorted(study_split):
        sp = study_split[sid]
        r = idx.get(sid)
        if r is None:
            raise SystemExit(f'{sid} 不在几何扫描（{scan_path}）里')
        if r['skip']:
            raise SystemExit(f'{sid} 在几何扫描里 skip=True —— 先查为什么')
        nw = int(r['n_windows'])
        if nw <= 0:
            raise SystemExit(f'{sid} 的 n_windows={nw}，不是正数')
        entries.append((sp, sid, nw))
        by_split[sp] = by_split.get(sp, 0) + 1
    return entries, by_split, len(scan['rows']), (n_train, n_test)


def _lpt(entries: list[tuple[str, str, int]], ngpu: int) -> list[list[tuple[str, str, int]]]:
    """按 n_windows 降序贪心投给当前最轻的片（同负载取小下标 ⇒ 同输入同切分）。"""
    order = sorted(entries, key=lambda e: (-e[2], e[0], e[1]))
    shards: list[list[tuple[str, str, int]]] = [[] for _ in range(ngpu)]
    load = [0] * ngpu
    for e in order:
        k = min(range(ngpu), key=lambda i: (load[i], i))
        shards[k].append(e)
        load[k] += e[2]
    return shards


def _shards_ok(out: str, ngpu: int, entries: list[tuple[str, str, int]]) -> tuple[bool, str]:
    d = os.path.join(out, 'shards')
    mark = os.path.join(d, '.ngpu')
    if not os.path.exists(mark):
        return False, '缺 .ngpu 标记'
    got = open(mark, encoding='utf-8').read().strip()
    if got != str(ngpu):
        return False, f'.ngpu={got} != 本次 {ngpu}'
    if os.path.exists(os.path.join(d, f'part_{ngpu}.txt')):
        return False, f'存在 part_{ngpu}.txt（盘上片数多于 {ngpu}）'
    seen: set[str] = set()
    for k in range(ngpu):
        p = os.path.join(d, f'part_{k}.txt')
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False, f'part_{k}.txt 缺失或为空'
        with open(p, encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    seen.add(ln)
    want = {f'{s}/{sid}' for s, sid, _ in entries}
    if seen != want:
        return False, f'片内容与当前数据不一致（盘上 {len(seen)} 条，现算 {len(want)} 条，异 {len(seen ^ want)}）'
    return True, 'ok'


def _source_hashes(train_json: str, eval_json: str, scan_path: str) -> dict[str, str]:
    src: dict[str, str] = {}
    for p in (train_json, eval_json, scan_path):
        ap = os.path.abspath(p)
        if not os.path.isfile(ap):
            raise SystemExit(f'plan 源文件不存在：{ap}')
        src[ap] = _sha256(ap)
    for p in KEEP_TSVS:
        ap = os.path.abspath(p)
        if os.path.isfile(ap):
            src[ap] = _sha256(ap)
    return src


def cmd_plan(args) -> int:
    out, ngpu = args.out, args.ngpu
    t0 = time.time()
    entries, by_split, n_scan, (n_train_rows, n_test_rows) = _collect(
        args.scan, args.train_json, args.eval_json)
    n_total = len(entries)
    win_total = sum(e[2] for e in entries)
    extra = n_scan - n_total
    print(f'[cache/plan] encoder={args.encoder}（a17：只决定 shard/ 与 plan.json 的口径，'
          f'数值身份在 run 里核对）')
    print(f'[cache/plan] 终版 JSON 现算 {n_total} 卷（' +
          ' '.join(f'{s} {n}' for s, n in by_split.items()) +
          f'；train_val {n_train_rows} 行 / test {n_test_rows} 行）；'
          f'几何扫描 {n_scan} 行（多 {extra} 个无记录的图片文件）')
    print(f'[cache/plan] n_windows 上界合计 {win_total}（crop 之后实测值以片尾汇总为准）')

    ok, why = _shards_ok(out, ngpu, entries)
    if ok:
        print(f'[cache/plan] 复用现有切分 shards/（.ngpu={ngpu}，{n_total} 条逐条一致）')
        shards = None
        pp = os.path.join(out, 'shards', 'plan.json')
        if os.path.exists(pp):
            with open(pp, encoding='utf-8') as f:
                cur_enc = json.load(f).get('encoder')
            if cur_enc != args.encoder:
                raise SystemExit(f'a17 拒绝续跑：{pp} 的 encoder={cur_enc!r} != 本次 '
                                 f'{args.encoder!r} —— 切分与编码器不同源，换 --out 或先归档')
    else:
        print(f'[cache/plan] 重切：{why}')
        shards = _lpt(entries, ngpu)
        d = os.path.join(out, 'shards')
        if os.path.isdir(d):
            stamp = time.strftime('%Y%m%d%H%M%S')
            dst = os.path.join(out, f'shards.stale-{stamp}')
            i = 1
            while os.path.exists(dst):
                dst = os.path.join(out, f'shards.stale-{stamp}.{i}')
                i += 1
            os.replace(d, dst)
            print(f'[cache/plan] 旧切分挪走（不删）-> {os.path.relpath(dst, out)}/')
        os.makedirs(d, exist_ok=True)
        for k, part in enumerate(shards):
            with open(os.path.join(d, f'part_{k}.txt'), 'w', encoding='utf-8') as f:
                for s, sid, _ in sorted(part):
                    f.write(f'{s}/{sid}\n')
        with open(os.path.join(d, '.ngpu'), 'w', encoding='utf-8') as f:
            f.write(f'{ngpu}\n')
        plan = {
            'encoder': args.encoder,
            'ckpt': os.path.basename(CKPT_BY_IDENTITY[args.encoder]),
            'ngpu': ngpu,
            'n_studies': n_total,
            'n_train_rows': n_train_rows,
            'n_test_rows': n_test_rows,
            'n_windows_upper': win_total,
            'by_split': by_split,
            'by_split_windows': {
                s: sum(e[2] for e in entries if e[0] == s) for s in config.SPLITS},
            'shards': [
                {'k': k, 'n_studies': len(p), 'n_windows_upper': sum(e[2] for e in p),
                 'by_split': {s: sum(1 for e in p if e[0] == s) for s in config.SPLITS}}
                for k, p in enumerate(shards)],
            'sources': _source_hashes(args.train_json, args.eval_json, args.scan),
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        tmp = os.path.join(d, '.plan.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(d, 'plan.json'))

    d = os.path.join(out, 'shards')
    n_lines = 0
    for k in range(ngpu):
        with open(os.path.join(d, f'part_{k}.txt'), encoding='utf-8') as f:
            cnt = sum(1 for ln in f if ln.strip())
        win = 0
        if shards is not None:
            win = sum(e[2] for e in shards[k])
        n_lines += cnt
        print(f'[cache/plan]   片 {k}: {cnt} 例' + (f'（窗口上界 {win}）' if win else ''))
    if n_lines != n_total:
        raise SystemExit(f'片行数合计 {n_lines} != 现算 {n_total} —— 切分与数据不同源')
    print(f'[cache/plan] Σ 片行数 == 现算 {n_total}；耗时 {time.time() - t0:.1f}s，'
          f'GPU 无关（plan 纯 CPU）')
    return 0



def _ensure_meta(meta_path: str, payload: dict) -> None:
    """a15：身份不一致就拒绝续跑；不存在则独占创建（多 worker 并发起跑也安全）。"""
    if os.path.exists(meta_path):
        with open(meta_path, encoding='utf-8') as f:
            cur = json.load(f)
        for k in ('encoder_identity', 'ckpt_sha256', 'n_tensors_visual', 'organs',
                  'prob_items', 'probs_pending', 'text_emb_sha256', 'has_whole'):
            if cur.get(k) != payload.get(k):
                raise SystemExit(
                    f'a15 拒绝续跑：{meta_path} 的 {k} 与本次加载不一致\n'
                    f'  盘上: {cur.get(k)}\n  本次: {payload.get(k)}\n'
                    '  缓存的列语义变了 —— 换输出根目录或先归档旧目录，不要混写')
        if cur.get('ckpt_path') != payload['ckpt_path']:
            print(f'[cache] ⚠️ meta 的 ckpt_path 不同（{cur.get("ckpt_path")} vs '
                  f'{payload["ckpt_path"]}）但 sha256 相同 —— 同一份权重的另一份拷贝，继续')
        if cur.get('text_emb_path') != payload['text_emb_path']:
            print(f'[cache] ⚠️ meta 的 text_emb_path 不同但 sha256 相同 —— 继续')
        return
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    tmp = meta_path + f'.tmp{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    try:
        os.link(tmp, meta_path)
    except FileExistsError:
        with open(meta_path, encoding='utf-8') as f:
            cur = json.load(f)
        bad = [k for k in payload if cur.get(k) != payload.get(k)]
        if bad:
            raise SystemExit(
                f'a15 拒绝续跑：meta.json 被并发写入了不同的 ({", ".join(bad)})')
    finally:
        os.unlink(tmp)


def _save_npz(dst: str, res: dict) -> None:
    """列的写法在 `visual_frontend.save_npz` —— 预提取与现算必须产出同一份列语义。"""
    VF.save_npz(dst, res)


def _cache_dir(out: str) -> str:
    """缓存根 -> 唯一的编码器目录（只认带 organs/encoder_identity 的缓存 meta）。"""
    if not os.path.isdir(out):
        raise SystemExit(f'缓存根目录不存在：{out}')
    cands = []
    for d in sorted(os.listdir(out)):
        mp = os.path.join(out, d, 'meta.json')
        if not os.path.isfile(mp):
            continue
        try:
            with open(mp, encoding='utf-8') as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(meta.get('organs'), list) and meta['organs'] \
                and 'encoder_identity' in meta:
            cands.append(d)
    if len(cands) != 1:
        raise SystemExit(f'{out} 下按 meta.json 认出的编码器目录有 {len(cands)} 个：{cands}'
                         ' —— 必须恰好一个')
    return os.path.join(out, cands[0])


def cmd_check(args) -> int:
    """零 GPU 校验 plan / shards / npz 与终版数据面是否同一份。"""
    out = args.out
    plan_path = os.path.join(out, 'shards', 'plan.json')
    if not os.path.isfile(plan_path):
        raise SystemExit(f'缺 {plan_path} —— 先跑 cache plan')
    with open(plan_path, encoding='utf-8') as f:
        plan = json.load(f)
    if plan.get('encoder') != args.encoder:
        raise SystemExit(f'plan.encoder={plan.get("encoder")!r} != {args.encoder!r}')
    if plan.get('ckpt') != os.path.basename(CKPT_BY_IDENTITY[args.encoder]):
        raise SystemExit(f'plan.ckpt={plan.get("ckpt")!r} != '
                         f'{os.path.basename(CKPT_BY_IDENTITY[args.encoder])!r}')
    src = plan.get('sources')
    if not isinstance(src, dict) or not src or not all(os.path.isabs(p) for p in src):
        raise SystemExit('plan.sources 不是终版数据口径（旧 out_v3 plan）—— 先 cache plan 重建')

    for p, sha in sorted(src.items()):
        if not os.path.isfile(p):
            raise SystemExit(f'plan 记录的源文件已不存在：{p}')
        got = _sha256(p)
        if got != sha:
            raise SystemExit(f'plan 源文件已变：{p}\n  盘上 {got}\n  plan {sha}')

    entries, by_split, _, (n_tr, n_te) = _collect(
        args.scan, args.train_json, args.eval_json)
    if plan.get('n_studies') != len(entries):
        raise SystemExit(f'plan.n_studies={plan.get("n_studies")} != 现算 {len(entries)}')
    if plan.get('by_split') != by_split:
        raise SystemExit(f'plan.by_split={plan.get("by_split")} != 现算 {by_split}')
    if 'n_train_rows' in plan and plan['n_train_rows'] != n_tr:
        raise SystemExit(f'plan.n_train_rows={plan["n_train_rows"]} != 现算 {n_tr}')
    if 'n_test_rows' in plan and plan['n_test_rows'] != n_te:
        raise SystemExit(f'plan.n_test_rows={plan["n_test_rows"]} != 现算 {n_te}')
    win = sum(nw for _, _, nw in entries)
    if plan.get('n_windows_upper') != win:
        raise SystemExit(f'plan.n_windows_upper={plan.get("n_windows_upper")} != 现算 {win}')

    ngpu = int(plan.get('ngpu') or 0)
    if ngpu < 1:
        raise SystemExit(f'plan.ngpu={plan.get("ngpu")!r} 不是正数')
    ok, why = _shards_ok(out, ngpu, entries)
    if not ok:
        raise SystemExit(f'shards 与 plan/终版数据不一致：{why}')

    root = _cache_dir(out)
    with open(os.path.join(root, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    if meta.get('encoder_identity') != args.encoder:
        raise SystemExit(f'{root}/meta.json 的 encoder_identity='
                         f'{meta.get("encoder_identity")!r} != {args.encoder!r}')

    want = {f'{sp}/{sid}' for sp, sid, _ in entries}
    missing = sorted(k for k in want
                     if not os.path.isfile(os.path.join(root, *k.split('/')) + '.npz'))
    if missing:
        raise SystemExit(f'缺 {len(missing)} 个 npz（例：{missing[:5]}）—— '
                         f'先补编码（当前 {root}）')
    have = set()
    for sp in config.SPLITS:
        d = os.path.join(root, sp)
        if os.path.isdir(d):
            have.update(f'{sp}/{f[:-4]}' for f in os.listdir(d) if f.endswith('.npz'))
    extra = sorted(have - want)

    print(f'[cache/check] plan / shards / npz 一致：{len(entries)} 卷（' +
          ' '.join(f'{s} {n}' for s, n in by_split.items()) +
          f'），{ngpu} 片，窗口上界 {win}')
    print(f'[cache/check] npz 命中 {len(want)} / 缺失 0；另有多余旧 npz {len(extra)} 个'
          '（不在 plan，不参与训练）')
    return 0


def cmd_run(args) -> int:
    import torch

    from radev.train import volume

    out, ngpu, shard = args.out, args.ngpu, args.shard
    if not (0 <= shard < ngpu):
        raise SystemExit(f'--shard {shard} 不在 0..{ngpu - 1}')
    tag = args.tag
    if tag and not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise SystemExit(f'--tag {tag!r} 只允许字母、数字、下划线、点和短横线')
    part_path = args.part_file or os.path.join(out, 'shards', f'part_{shard}.txt')
    if not os.path.exists(part_path):
        raise SystemExit(f'缺 {part_path} —— 先跑 `cache plan`，或用 --part-file 指一份现成清单')
    with open(part_path, encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if args.limit:
        lines = lines[:args.limit]

    t0 = time.time()
    enc = args.encoder
    ident = RE.ENCODER_IDENTITIES[enc]
    ckpt = args.ckpt or CKPT_BY_IDENTITY[enc]
    cls = getattr(RE, ident['cls'])
    model, info = cls.from_pretrained(ckpt, identity=enc, device=args.device)
    if info['n_tensors_visual'] != ident['n_tensors']:
        raise SystemExit(f"视觉张量 {info['n_tensors_visual']} != 身份 {enc} 的 "
                         f"{ident['n_tensors']}（a17）")

    probs_pending = ident['text_lang'] != 'zh'
    text_emb_path = None
    td = {}
    text_emb: dict = {}
    if not probs_pending:
        text_emb_path = args.text_emb or DEFAULT_TEXT_EMB
        td = torch.load(text_emb_path, map_location='cpu', weights_only=True)
        miss = [it for it in RE.PROB_ITEMS if it not in td]
        if miss:
            raise SystemExit(f'文本嵌入缺 {len(miss)} 个征象键：{miss}')
        text_emb = {k: v.to(args.device) for k, v in td.items()}

    sha12 = info['ckpt_sha256'][:12]
    root = os.path.join(out, sha12)
    meta_payload = {
        'encoder_identity': enc,
        'ckpt_path': info['ckpt_path'],
        'ckpt_sha256': info['ckpt_sha256'],
        'n_tensors_visual': info['n_tensors_visual'],
        'organs': list(RE.ORGANS),
        'prob_items': [] if probs_pending else list(RE.PROB_ITEMS),
        'probs_pending': probs_pending,
        'probs_pending_items': list(RE.PROB_ITEMS) if probs_pending else [],
        'has_whole': ident['has_whole'],
        'roi_size': list(RE.ROI_SIZE),
        'overlap': RE.OVERLAP,
        'text_emb_path': None if text_emb_path is None else os.path.abspath(text_emb_path),
        'text_emb_sha256': None if text_emb_path is None else _sha256(text_emb_path),
        'text_emb_keys': sorted(td.keys()),
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'code': 'radev.train.cache',
    }
    _ensure_meta(os.path.join(root, 'meta.json'), meta_payload)

    logs = os.path.join(out, 'logs')
    os.makedirs(logs, exist_ok=True)
    fail_path = os.path.join(logs, f'failed_shard{shard}{tag}.txt')
    skip_path = os.path.join(logs, f'skipped_gt1000_shard{shard}{tag}.txt')
    whole_fail_path = os.path.join(logs, f'whole_failed_shard{shard}{tag}.txt')

    pad_func = RE.default_pad_func()
    n_done = n_skip_exist = n_failed = n_gt1000 = 0
    n_whole = n_whole_failed = 0
    win_actual = 0
    print(f'[cache/run] shard {shard}/{ngpu}：{len(lines)} 例；device={args.device}；'
          f'root={os.path.relpath(root, _REPO)}/；encoder={enc} '
          f'ckpt={os.path.basename(ckpt)} sha256={sha12}…', flush=True)
    if probs_pending:
        print('[cache/run] ⚠️ 本支不产 probs（英文文本塔不在本步）：npz 的 probs/prob_keys '
              '是空数组，20 项概率由离线重算补（任务 #60）', flush=True)
    for i, line in enumerate(lines, 1):
        split, sid = line.split('/', 1)
        dst = os.path.join(root, split, f'{sid}.npz')
        if os.path.exists(dst) and not args.overwrite:
            n_skip_exist += 1
            continue
        try:
            img = config.image_path(sid)
            image, _meta = volume.load_and_prepare(img)
            if any(int(s) > 1000 for s in image.shape[1:]):
                _append(skip_path, line)
                n_gt1000 += 1
                print(f'[cache/run] ⚠️ {line} padded {tuple(image.shape[1:])} 有维 >1000，'
                      '按上游口径跳过', flush=True)
                continue
            res = RE.infer_volume(
                model, image[None], RE.TEST_ITEMS if not probs_pending else [], text_emb,
                device=args.device, pad_func=pad_func, compute_organs=RE.ORGANS)
            del image, _meta
            res['whole_feat'] = None
            if ident['has_whole']:
                try:
                    wimg, _ = volume.load_and_prepare_whole(img)
                    res['whole_feat'] = RE.infer_whole(model, wimg[None], device=args.device)
                    n_whole += 1
                except Exception as e:
                    _append(whole_fail_path, f'{line}\t{type(e).__name__}: {e}')
                    n_whole_failed += 1
                    print(f'[cache/run] ⚠️ {line} 整图支路失败（has_whole=False）：'
                          f'{type(e).__name__}: {e}', flush=True)
            _save_npz(dst, res)
            n_done += 1
            win_actual += int(res['n_windows'])
        except Exception as e:
            _append(fail_path, f'{line}\t{type(e).__name__}: {e}')
            n_failed += 1
            print(f'[cache/run] ✗ {line}: {type(e).__name__}: {e}', flush=True)
        if i % args.log_every == 0:
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(lines) - i) / rate if rate else 0
            print(f'[cache/run] shard {shard}: {i}/{len(lines)} 例 '
                  f'（新写 {n_done} / 已有 {n_skip_exist} / 失败 {n_failed}）；'
                  f'{rate:.2f} 例/s，已用 {el / 60:.1f} min，ETA {eta / 60:.1f} min', flush=True)

    summary = {
        'shard': shard, 'ngpu': ngpu, 'device': args.device,
        'encoder_identity': enc,
        'part_file': part_path, 'tag': tag,
        'n_total': len(lines), 'n_done': n_done, 'n_skipped_existing': n_skip_exist,
        'n_failed': n_failed, 'n_skipped_gt1000': n_gt1000,
        'n_whole': n_whole, 'n_whole_failed': n_whole_failed,
        'probs_pending': probs_pending,
        'n_windows_actual': win_actual,
        'ckpt_sha256': info['ckpt_sha256'],
        'elapsed_sec': round(time.time() - t0, 1),
        'finished_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(logs, f'summary_shard{shard}{tag}.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f'[cache/run] shard {shard} 完成：新写 {n_done} / 已有 {n_skip_exist} / '
          f'失败 {n_failed} / >1000 跳过 {n_gt1000}；整图向量 {n_whole} 例有 / '
          f'{n_whole_failed} 例无；实际滑窗 {win_actual}；{summary["elapsed_sec"]}s',
          flush=True)
    return 0



def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='python -m radev.train.cache',
                                description='stage 1：RADAR 冻结编码器逐 study 缓存'
                                            '（也可从 `python -m radev radar-cache` 透传进来）')
    sub = p.add_subparsers(dest='cmd', required=True)

    pp = sub.add_parser('plan', help='切分（LPT 按 n_windows 均衡；幂等，复用前逐条核对）')
    pp.add_argument('--out', default=DEFAULT_OUT,
                    help=f'缓存根目录（默认 {os.path.basename(DEFAULT_OUT)}/）')
    pp.add_argument('--scan', default=DEFAULT_SCAN,
                    help='geometry_scan.json（与编码器无关，两支共用）')
    pp.add_argument('--train-json', default=DEFAULT_TRAIN_JSON, dest='train_json',
                    help='终版训练集（默认 out_dataset_final_sym/train_val.json）')
    pp.add_argument('--eval-json', default=DEFAULT_EVAL_JSON, dest='eval_json',
                    help='终版测试集（默认 out_dataset_final_sym/test.json）')
    pp.add_argument('--encoder', default=DEFAULT_ENCODER, choices=sorted(RE.ENCODER_IDENTITIES),
                    dest='encoder', help=f'a17 声明用哪一支；默认 {DEFAULT_ENCODER}')
    pp.add_argument('--ngpu', type=int, default=8, help='片数；一个 worker 一片')
    pp.set_defaults(fn=cmd_plan)

    pc = sub.add_parser('check', help='零 GPU 校验：plan / shards / npz 与终版数据一致')
    pc.add_argument('--out', default=DEFAULT_OUT)
    pc.add_argument('--scan', default=DEFAULT_SCAN)
    pc.add_argument('--train-json', default=DEFAULT_TRAIN_JSON, dest='train_json')
    pc.add_argument('--eval-json', default=DEFAULT_EVAL_JSON, dest='eval_json')
    pc.add_argument('--encoder', default=DEFAULT_ENCODER, choices=sorted(RE.ENCODER_IDENTITIES))
    pc.set_defaults(fn=cmd_check)

    pr = sub.add_parser('run', help='跑第 --shard 片（真跑，需要 GPU 或 --device cpu 冒烟）')
    pr.add_argument('--shard', type=int, required=True)
    pr.add_argument('--ngpu', type=int, required=True)
    pr.add_argument('--out', default=DEFAULT_OUT)
    pr.add_argument('--encoder', default=DEFAULT_ENCODER, choices=sorted(RE.ENCODER_IDENTITIES),
                    help=f'a17：声明编码器身份；默认 {DEFAULT_ENCODER}')
    pr.add_argument('--ckpt', default=None,
                    help='默认按 --encoder 取 CKPT_BY_IDENTITY（给了就两样一起核对）')
    pr.add_argument('--text-emb', default=None, dest='text_emb',
                    help=f'仅 pretrain 支用；默认 {DEFAULT_TEXT_EMB}')
    pr.add_argument('--device', default='cuda', help="cuda（默认）或 cpu")
    pr.add_argument('--part-file', default=None, dest='part_file',
                    help='自定义逐例清单（每行 split/sid）；默认 <out>/shards/part_<shard>.txt')
    pr.add_argument('--tag', default='',
                    help='summary/failed/skipped 文件名后缀（例如 _reverse，避免与另一条 run 重名）')
    pr.add_argument('--limit', type=int, default=0, help='只跑本片前 N 例（冒烟用）')
    pr.add_argument('--overwrite', action='store_true', help='已有的 npz 也重算')
    pr.add_argument('--log-every', type=int, default=100, dest='log_every')
    pr.set_defaults(fn=cmd_run)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == '__main__':
    sys.exit(main())
