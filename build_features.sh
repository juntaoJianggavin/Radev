#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "[radar-cache] FATAL: 第 $LINENO 行 \`$BASH_COMMAND\` 非零退出（rc=$rc）—— 这不是判据，是脚本被外部命令打死" >&2; exit $rc' ERR

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python3}"
ENCODER="${ENCODER:-radar_ft}"
OUT="${OUT:-$REPO/feature_cache}"
SCAN="${SCAN:-$REPO/geometry_scan.json}"
CKPT="${CKPT:-}"
TEXT_EMB="${TEXT_EMB:-}"
NGPU="${NGPU:-8}"
GO="${GO:-0}"
DEVICE="${DEVICE:-cuda}"
OMP_THREADS="${OMP_THREADS:-12}"
FREE_MIN_MIB="${FREE_MIN_MIB:-4096}"
LIMIT="${LIMIT:-0}"
OVERWRITE="${OVERWRITE:-0}"

LIMIT_ARG=""; [ "$LIMIT" != "0" ] && LIMIT_ARG="--limit $LIMIT"
OVERWRITE_ARG=""; [ "$OVERWRITE" = "1" ] && OVERWRITE_ARG="--overwrite"
CKPT_ARG=""; [ -n "$CKPT" ] && CKPT_ARG="--ckpt $CKPT"
TEXT_ARG=""; [ -n "$TEXT_EMB" ] && TEXT_ARG="--text-emb $TEXT_EMB"

log() { echo "[radar-cache] $*"; }

cd "$REPO" || { log "FATAL: 进不了 REPO=$REPO"; exit 1; }

"$PY" - <<'PYEOF' || { log "FATAL: $PY 缺 torch/monai/numpy —— 用 PY=/path 指到正确的 env"; exit 1; }
import torch, monai, numpy
print(f"[radar-cache] 解释器自检 ok：torch {torch.__version__} / monai {monai.__version__}"
      f" / numpy {numpy.__version__} / cuda_available={torch.cuda.is_available()}")
PYEOF

log "stage 1 切分：plan --ngpu $NGPU --encoder $ENCODER（复用前逐条核对；对不上才重切）"
"$PY" -m radev.train.cache plan --out "$OUT" --ngpu "$NGPU" \
  --encoder "$ENCODER" --scan "$SCAN"

if [ "$GO" != "1" ]; then
  log "DRY：一个 GPU 都没碰。要真跑：GO=1 bash $0（8 卡；本机单卡冒烟 GO=1 DEVICE=cpu LIMIT=2 NGPU=1）"
  exit 0
fi

if [ "$DEVICE" != "cpu" ]; then
  n_card=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ') || n_card=0
  if [ "${n_card:-0}" -lt "$NGPU" ]; then
    log "FATAL: 这台机器可见 ${n_card:-0} 张卡，但 NGPU=$NGPU —— 不起 worker、不搬任何东西。" >&2
    exit 1
  fi
  mapfile -t free_mib < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  bad=()
  for ((k = 0; k < NGPU; k++)); do
    f="${free_mib[$k]:-0}"; f="${f// /}"
    [[ "$f" =~ ^[0-9]+$ ]] || f=0
    if [ "$f" -lt "$FREE_MIN_MIB" ]; then
      bad+=("$k(空闲 ${f} MiB)")
    fi
    log "  卡 $k：空闲 $((f / 1024)) GiB（门槛 $((FREE_MIN_MIB / 1024)) GiB）"
  done
  if [ "${#bad[@]}" -gt 0 ]; then
    log "FATAL: 这些卡空闲不够 —— ${bad[*]}。当前占用者：" >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' >&2 || true
    log "  不挤掉别人的进程。要么等，要么换节点，要么 NGPU 调小到够用的卡数。" >&2
    exit 1
  fi
else
  log "DEVICE=cpu：跳过 GPU 预检（冒烟路径）"
fi

mkdir -p "$OUT/logs"
pids=(); shards=()
for ((k = 0; k < NGPU; k++)); do
  log "起 worker：shard $k/$NGPU -> logs/shard_$k.log"
  CUDA_VISIBLE_DEVICES="$k" OMP_NUM_THREADS="$OMP_THREADS" \
    "$PY" -m radev.train.cache run --shard "$k" --ngpu "$NGPU" --out "$OUT" \
    --encoder "$ENCODER" --device "$DEVICE" \
    $CKPT_ARG $TEXT_ARG $LIMIT_ARG $OVERWRITE_ARG > "$OUT/logs/shard_$k.log" 2>&1 &
  pids+=($!); shards+=("$k")
done

rc_all=0
for i in "${!pids[@]}"; do
  k="${shards[$i]}"
  if wait "${pids[$i]}"; then
    log "worker shard $k 成功"
  else
    rc=$?
    rc_all=1
    log "🔴 worker shard $k 的 rc=$rc —— 见 logs/shard_$k.log 尾部" >&2
    tail -n 5 "$OUT/logs/shard_$k.log" | sed 's/^/    /' >&2 || true
  fi
done

rc=0
"$PY" - "$OUT" "$LIMIT" "$ENCODER" <<'PYEOF' || rc="$?"
import json, os, sys
out, limit, enc = sys.argv[1], int(sys.argv[2]), sys.argv[3]
plan = json.load(open(os.path.join(out, 'shards', 'plan.json'), encoding='utf-8'))
if plan.get('encoder') != enc:
    sys.exit(f'[radar-cache] FATAL: shards/plan.json 的 encoder={plan.get("encoder")!r} '
             f'!= 本次 ENCODER={enc!r} —— 切分与编码器不同源（a17）')
metas = []
for name in sorted(os.listdir(out)):
    p = os.path.join(out, name, 'meta.json')
    if len(name) == 12 and os.path.exists(p):
        metas.append((name, json.load(open(p, encoding='utf-8'))))
if len(metas) != 1:
    sys.exit(f'[radar-cache] FATAL: {out} 下有 {len(metas)} 个带 meta.json 的 sha12 目录 '
             f'{[m[0] for m in metas]} —— 不能对账（换过 encoder 就先归档旧目录）')
sha12, meta = metas[0]
got_enc = meta.get('encoder_identity')
if got_enc != enc:
    sys.exit(f'[radar-cache] FATAL: meta.json 的 encoder_identity={got_enc!r} != {enc!r}（a15）')
if os.path.basename(meta['ckpt_path']) != plan.get('ckpt'):
    sys.exit(f'[radar-cache] FATAL: meta 的权重 {os.path.basename(meta["ckpt_path"])} '
             f'与 plan 声明的 {plan.get("ckpt")} 不是同一支')
print(f'[radar-cache] 对账 encoder={enc} sha12={sha12}'
      f'（{os.path.basename(meta["ckpt_path"])}，视觉张量 {meta.get("n_tensors_visual")}）'
      f' plan.ngpu={plan["ngpu"]}')
if meta.get('probs_pending'):
    n = len(meta.get('probs_pending_items') or [])
    print(f'[radar-cache] ⚠️ 这一支不产概率（文本塔不同源）：probs 列空，'
          f'待离线重算的 {n} 项见任务 #60 —— ② 级地板在此之前不能报')
ok = True
total_disk = total_plan = 0
for split, exp in sorted(plan['by_split'].items()):
    d = os.path.join(out, sha12, split)
    got = sum(1 for f in os.listdir(d) if f.endswith('.npz')) if os.path.isdir(d) else 0
    total_disk += got; total_plan += exp
    mark = 'OK' if got == exp else ('缺' if got < exp else '多')
    if got != exp:
        ok = False
    print(f'[radar-cache]   {split}: 盘上 {got} / 计划 {exp} [{mark}]')
if limit:
    print(f'[radar-cache] ⚠️ 本轮带 LIMIT={limit}（冒烟），盘上必然少于计划 —— 不作数')
    sys.exit(0)
if not ok:
    print(f'[radar-cache] 🔴 未完成：{total_disk}/{total_plan} —— 重跑同一条命令即续跑'
          f'（已有 npz 跳过；失败的看 logs/failed_shard*.txt）')
    sys.exit(1)
win = n_whole = n_whole_fail = 0
for k in range(plan['ngpu']):
    p = os.path.join(out, 'logs', f'summary_shard{k}.json')
    if not os.path.exists(p):
        continue
    s = json.load(open(p, encoding='utf-8'))
    win += s.get('n_windows_actual', 0)
    n_whole += s.get('n_whole', 0)
    n_whole_fail += s.get('n_whole_failed', 0)
if meta.get('has_whole'):
    print(f'[radar-cache]   整图向量：{n_whole}/{total_disk} 例有，{n_whole_fail} 例失败'
          f'（失败的那部分 has_whole=False，projector 走可学 missing 槽）')
    if n_whole_fail:
        print(f'[radar-cache] ⚠️ 整图失败明细见 logs/whole_failed_shard*.txt —— '
              f'非有限值（归一化 0/0）是最可能的一类，见 volume.py 注释')
print(f'[radar-cache] ✅ 全部 {total_disk} 例落盘；实际滑窗 {win}（几何扫描上界 '
      f'{plan["n_windows_upper"]} —— crop 之后应当更小）')
PYEOF
if [ "$rc" != "0" ]; then exit "$rc"; fi
[ "$rc_all" != "0" ] && exit "$rc_all"
log "完成（NGPU=$NGPU，OUT=$OUT）"
