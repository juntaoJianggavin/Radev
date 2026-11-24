#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python3}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GO="${GO:-0}"
STAGE="${STAGE:-measure}"
NGPU="${NGPU:-4}"
GPUS="${GPUS:-0,1,2,3}"
CUDA_DEV="${CUDA_DEV:-0}"
MASTER_PORT="${MASTER_PORT:-29550}"

SIZE="${SIZE:-2B}"
LLM_ROOT="${LLM_ROOT:-$REPO/pretrained}"
VISUAL_ROOT="${VISUAL_ROOT:-$REPO/feature_cache}"
TRAIN_JSON="${TRAIN_JSON:-$REPO/data/iid/train_val.json}"
EVAL_JSON="${EVAL_JSON:-$REPO/data/iid/test.json}"
RULES="${RULES:-explicit_negative,asserted_positive,hedged_positive,implied_negative_v2}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train+val}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
ORGAN_MAP="${ORGAN_MAP:-$REPO/organ_map.tsv}"

SRC="${SRC:-$REPO/checkpoints/vft}"
SRC_CK="${SRC_CK:-$SRC/ckpt_last.pt}"
OUT_ROOT="${OUT_ROOT:-$REPO/checkpoints}"
SLICES="${SLICES:-32}"
SLICE_PROC_LRU="${SLICE_PROC_LRU:-8}"

LIMIT="${LIMIT:-700}"
M_STEPS="${M_STEPS:-20}"
BS="${BS:-1}"
ACCUM="${ACCUM:-8}"
LR="${LR:-4e-6}"
SEED="${SEED:-0}"
LOG_EVERY="${LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-500}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
EVAL_BS="${EVAL_BS:-8}"
EVAL_SHARDS="${EVAL_SHARDS:-$NGPU}"

VIEWS="${VIEWS:-4}"
LAM_RANK="${LAM_RANK:-0}"
LAM_NULL="${LAM_NULL:-0}"
M_RANK="${M_RANK:-0.05}"
M_NULL="${M_NULL:-0.03}"
TAU="${TAU:-0.5}"
MIN_FRAC="${MIN_FRAC:-0.005}"
MAX_REPEAT="${MAX_REPEAT:-8}"
MASK_WHOLE="${MASK_WHOLE:-1}"
TASK_VIEWS="${TASK_VIEWS:-all}"

BLIND="${BLIND:-1}"
LAM_BLIND="${LAM_BLIND:-0.2}"
M_BLIND="${M_BLIND:-0.20}"

_tag_set="${TAG:+1}"
TAG="${TAG:-cct_s32-mw1-tvall-lam0-bl1-mb${M_BLIND}-s${LIMIT}}"
OUT="${OUT:-$OUT_ROOT/$TAG}"

log() { printf '[racd] %s\n' "$*" >&2; }

[ -n "$SRC_CK" ] || { log "FATAL: SRC_CK 空"; exit 1; }
if [ "$GO" = "1" ] && [ ! -f "$SRC_CK" ]; then
  log "FATAL: 基座 ckpt 不在：$SRC_CK"
  log "       换 SRC=<run 目录> 或 SRC_CK=<ckpt 路径>；先确认那一臂真跑完了"
  exit 1
fi
[ -f "$TRAIN_JSON" ] || { log "FATAL: 缺 --train-json $TRAIN_JSON"; exit 1; }
[ -f "$ORGAN_MAP" ] || { log "FATAL: 缺器官表 $ORGAN_MAP（视图与分层采样都靠它）"; exit 1; }
case "$STAGE" in measure|train|eval|both) ;; *) log "FATAL: STAGE=$STAGE 应为 measure|train|eval|both"; exit 1 ;; esac
case "$MASK_WHOLE" in 0|1) ;; *) log "FATAL: MASK_WHOLE=$MASK_WHOLE 应为 0|1"; exit 1 ;; esac
case "$TASK_VIEWS" in all|no_target|non_gated) ;; *)
  log "FATAL: TASK_VIEWS=$TASK_VIEWS 应为 all|no_target|non_gated"; exit 1 ;; esac
case "$BLIND" in 0|1) ;; *) log "FATAL: BLIND=$BLIND 应为 0|1"; exit 1 ;; esac
case "$VIEWS" in ''|*[!0-9]*) log "FATAL: VIEWS=$VIEWS 应为整数"; exit 1 ;; esac
if [ "$BLIND" = "1" ]; then
  if [ "$VIEWS" -lt 3 ]; then
    log "FATAL: BLIND=1 要 VIEWS>=3（收到 $VIEWS）—— 盲视图恒占最后一槽，"
    log "       剩下的至少要是 full 与 target 两端（views.plan_views 也会炸，这里先拦）"
    exit 1
  fi
  if [ "$MASK_WHOLE" != "1" ]; then
    log "FATAL: BLIND=1 要 MASK_WHOLE=1 —— 这条臂序列里有第 37 个整图槽（oev s32 基座），"
    log "       不遮它的话盲视图仍带 whole_feat ⇒ c_full − c_blind 量到的是**泄漏**，不是语言先验。"
    log "       那正是把 racd1 的 gate_frac 从 0.0431 压到 0.0194、让 λ_corr 空转的同一条通路（#191）。"
    exit 1
  fi
fi
_lbnz=$(awk -v x="$LAM_BLIND" 'BEGIN{printf "%d", (x+0 > 0)}')
_mbng=$(awk -v x="$M_BLIND" 'BEGIN{printf "%d", (x+0 < 0)}')
if [ "$BLIND" != "1" ] && [ "$_lbnz" = "1" ]; then
  log "FATAL: LAM_BLIND=$LAM_BLIND >0 但 BLIND=0 ⇒ 批次里一个 blind 槽都没有，这一项恒 0。"
  log "       那是一条「什么也没训」的废臂（前向不报错、曲线照样出），与 λ_corr=0 的 GRPO 臂同类。"
  exit 1
fi
if [ "$_mbng" = "1" ]; then
  log "FATAL: M_BLIND=$M_BLIND < 0 —— 间隔为负等于要求盲视图比整卷**更**自信（翻号）"; exit 1
fi
_mbhi=$(awk -v x="$M_BLIND" 'BEGIN{printf "%d", (x+0 > 0.40)}')
if [ "$_mbhi" = "1" ] && [ "$BLIND" = "1" ]; then
  log "⚠ M_BLIND=$M_BLIND > 0.40：noul（K=2，c ≥ 0.5）的 d_blind 上界只有 c_full − 0.5 ≈ 0.47，"
  log "  要求 >0.40 在 noul 上近乎不可满足 ⇒ hinge 变成**无条件罚**，失去「只在真违约处用力」的选择性。"
  log "  风险类别与 RAC 在 λ_corr=0.4 那档 acc 塌到 0.589 同类。除非 measure 段的 CDF 明确要求，别这么设。"
fi
case "$EVAL_SHARDS" in ''|*[!0-9]*) log "FATAL: EVAL_SHARDS=$EVAL_SHARDS 应为整数（1=单卡原路径）"; exit 1 ;; esac
case "$SLICE_PROC_LRU" in ''|*[!0-9]*) log "FATAL: SLICE_PROC_LRU=$SLICE_PROC_LRU 应为非负整数（0=关，逐位老行为）"; exit 1 ;; esac
[ "$EVAL_SHARDS" -ge 1 ] || { log "FATAL: EVAL_SHARDS=$EVAL_SHARDS < 1"; exit 1; }
IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [ "$GO" = "1" ] && [ "$NGPU" -gt 1 ]; then
  [ "${#GPU_ARR[@]}" -ge "$NGPU" ] || { log "FATAL: GPUS=$GPUS 只有 ${#GPU_ARR[@]} 张，不够 NGPU=$NGPU"; exit 1; }
fi
if [ "$EVAL_SHARDS" -gt 1 ] && [ "${#GPU_ARR[@]}" -lt "$EVAL_SHARDS" ]; then
  log "FATAL: EVAL_SHARDS=$EVAL_SHARDS 需要 $EVAL_SHARDS 张卡，GPUS=$GPUS 只有 ${#GPU_ARR[@]} 张"
  log "       （要单卡口径就设 EVAL_SHARDS=1；report 由最后那次 --merge-shards 出）"
  exit 1
fi

ARM_ARGS=(--feats cache --encoder radar_ft --train-split "$TRAIN_SPLIT"
          --rules "$RULES" --dtype bfloat16 --bs "$BS" --accum "$ACCUM" --seed "$SEED"
          --loss racd --slices "$SLICES" --slice-root "$REPO/slice_cache" --slice-lru 8
          --slice-proc-lru "$SLICE_PROC_LRU"
          --llm "$LLM_ROOT/Qwen3.5-$SIZE" --organ-evidence --prompt-no-count
          --cache-root "$VISUAL_ROOT" --organ-map "$ORGAN_MAP"
          --train-json "$TRAIN_JSON" --eval-json "$EVAL_JSON")
RACD_ARGS=(--racd-views "$VIEWS" --racd-lam-rank "$LAM_RANK" --racd-lam-null "$LAM_NULL"
           --racd-m-rank "$M_RANK" --racd-m-null "$M_NULL"
           --racd-tau "$TAU" --racd-min-frac "$MIN_FRAC" --racd-max-repeat "$MAX_REPEAT"
           --racd-task-views "$TASK_VIEWS"
           --racd-lam-blind "$LAM_BLIND" --racd-m-blind "$M_BLIND")
[ "$MASK_WHOLE" = "1" ] && RACD_ARGS+=(--racd-mask-whole)
[ "$BLIND" = "1" ] && RACD_ARGS+=(--racd-blind)
case "$TAG" in *mw*) _mw_mark=1 ;; *) _mw_mark=0 ;; esac
case "$TAG" in *tv*) _tv_mark=1 ;; *) _tv_mark=0 ;; esac
case "$TAG" in *bl*) _bl_mark=1 ;; *) _bl_mark=0 ;; esac
if [ "$BLIND" = "1" ] && [ -n "$_tag_set" ] && [ "$_bl_mark" = 0 ]; then
  log "⚠ 自定 TAG=$TAG 里没有 bl 标记，但 BLIND=1 换的是**视图数**（M=$VIEWS 而不是 3）："
  log "  每步多 1/3 行前向、CE 行集多一个被摘掉的盲槽。与 VIEWS=3 的 run 混在一个目录里就分不开了。"
fi
if [ "$MASK_WHOLE" = "1" ] && [ -n "$_tag_set" ] && [ "$_mw_mark" = 0 ]; then
  log "⚠ 自定 TAG=$TAG 里没有 mw 标记，但 MASK_WHOLE=1 换的是反事实的**定义**（新臂）。"
  log "  产物会写进 $OUT —— 若那儿已经存过 MASK_WHOLE=0 的东西，两种口径就混在一个目录里了。"
fi
if [ "$TASK_VIEWS" != "all" ] && [ -n "$_tag_set" ] && [ "$_tv_mark" = 0 ]; then
  log "⚠ 自定 TAG=$TAG 里没有 tv 标记，但 TASK_VIEWS=$TASK_VIEWS 换的是答案监督的**行集**（新臂）。"
  log "  产物会写进 $OUT —— 与 all 口径的 run 混在一个目录里就分不开是谁的数了。"
fi

_launch() {
  if [ "$NGPU" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
      "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
      -m radev train train "$@"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
      "$PY" -u -m radev train train "$@"
  fi
}
_eval_dev() { if [ "$NGPU" -gt 1 ]; then echo "${GPUS%%,*}"; else echo "$CUDA_DEV"; fi; }

mkdir -p "$OUT/logs"
log "CCT 阶段 1 · 盲视图臂（run_cct.sh）：基座 $SRC_CK"
log "  out=$OUT ngpu=$NGPU steps=$LIMIT M=$VIEWS bs=$BS accum=$ACCUM lr=$LR"
log "  OMP_NUM_THREADS=$OMP_NUM_THREADS slice_proc_lru=$SLICE_PROC_LRU（提速口径，数值逐位不变：见 analysis/check_proc*_*.py）"
log "  λ_rank=$LAM_RANK λ_null=$LAM_NULL m_rank=$M_RANK m_null=$M_NULL"
if [ "$MASK_WHOLE" = "1" ]; then
  log "  反事实：槽1 = 屏蔽绑定器官槽 **+ 整图全局槽**（whole_feat 是泄漏通路；TAG 带 -mw1）"
else
  log "  反事实：槽1 = 只屏蔽绑定器官槽（整图槽豁免 = 已跑两条臂的口径；本臂实测 gate_frac 0.043→0.019）"
fi
if [ "$BLIND" = "1" ]; then
  log "  🔴 盲视图：槽$((VIEWS - 1)) = 屏蔽**全部**在场槽（含整图槽）= 「只看题面不看图」那一端"
  log "     λ_blind=$LAM_BLIND m_blind=$M_BLIND（要求 c_full − c_blind ≥ m_blind，门控 = 盲掉真把答案打坏了）"
  log "     c_full 在这一项里 **detach** ⇒ 对报数口径读的那个信心零直接梯度（不 detach 就是 13 条 GRPO 臂赔校准的方向）"
  log "     🔴 m_blind/λ_blind 是**暂定值**：先跑 STAGE=measure 拿六点 CDF bgap_lt_* 再定，改完换 TAG（-mb<新值>）"
else
  log "  盲视图：关（BLIND=0）⇒ 本副本退化成一条 VIEWS=$VIEWS 的普通 racd 臂"
fi
log "  分层 tau=$TAU min_frac=$MIN_FRAC max_repeat=$MAX_REPEAT"
if [ "$BLIND" = "1" ]; then
  log "  答案监督：盲视图槽**无条件退出 CE**（不看 TASK_VIEWS）⇒ task_keep ≈ $((VIEWS - 1))/$VIEWS"
  log "            吃 CE 的那 $((VIEWS - 1)) 行 = full/target/distractor，与 VIEWS=3 的 armD **逐位同构**"
fi
if [ "$TASK_VIEWS" = "all" ]; then
  log "  答案监督口径=$TASK_VIEWS：非盲槽全部吃 CE（armC/armD 的口径）"
elif [ "$TASK_VIEWS" = "no_target" ]; then
  log "  答案监督：target_mask 行**退出 CE**，只由 rank 塑形（RAC/RLCD 原式）"
else
  log "  答案监督：只摘门控为真的 target 行（遮蔽确实打坏答案的那几组让给 rank），其余照旧吃 CE"
fi
log "  🔴 预注册：不划 val、不 early stop、不选 ckpt；收敛只看 train_log.jsonl 的 racd 块"

if [ "$STAGE" = "measure" ]; then
  log "=== 计时 $M_STEPS 步（M=$VIEWS 视图 ⇒ 每步 $((BS * VIEWS)) 行前向）+ 定标 m_blind"
  log "    🔴 这一段不只是计时：它是 m_blind/λ_blind 的**唯一**定标依据（d_blind 的分布从没落过盘），"
  log "       也是 VIEWS=4 的**显存闸**（比 armD 多 1/3 行前向，会不会 OOM 在这 6 分钟里就知道）。"
  if [ "$GO" = "1" ]; then
    t0=$(date +%s)
    _launch "${ARM_ARGS[@]}" "${RACD_ARGS[@]}" --lr "$LR" --epochs 1 --limit "$M_STEPS" \
      --log-every 5 --init-from "$SRC_CK" --resume off \
      --out "$OUT/measure" > "$OUT/logs/measure.log" 2>&1
    el=$(( $(date +%s) - t0 ))
    tail -n 8 "$OUT/logs/measure.log" | sed 's/^/    /'
    awk -v e="$el" -v k="$M_STEPS" -v L="$LIMIT" -v ng="$NGPU" -v m="$VIEWS" -v b="$BS" 'BEGIN {
      printf "    [racd] => %.2f s/step（%d 卡，每题 %d 视图，全局 bs=%d）\n", e/k, ng, m, b*ng
      printf "    [racd]    %d 步 ≈ %.1f h（一个 SFT epoch 是 7053 步，这里 %.2f epoch）\n",
             L, e/k*L/3600, L/7053 }'
    if [ -f "$OUT/measure/train_log.jsonl" ]; then
      "$PY" - "$OUT/measure/train_log.jsonl" "$M_BLIND" "$LAM_BLIND" <<'PYEOF'
import json, sys
p, mb, lb = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
rows = [json.loads(l) for l in open(p) if l.strip()]
last = rows[-1]
r = last.get('racd') or last.get('diag') or last
def g(k, d=None):
    return r.get(k, d)
th = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60)
print(f'    [定标] step={last.get("step")}  n_blind={g("n_blind")}  n_blind_gate={g("n_blind_gate")}')
if g('n_blind') in (None, 0):
    print('    [定标] 🔴 一个盲槽都没配上 ⇒ --racd-blind 没生效或这批全退回 full；先查这个，别拿 CDF 定标')
    sys.exit(0)
cdf = [(t, g(f'bgap_lt_{f"{t:g}".replace(".", "p")}')) for t in th]
print('    [定标] d_blind = c_full − c_blind 的六点 CDF（权威口径 = bgap_lt_*，按 n_blind_gate 加权 = 池化）：')
for t, v in cdf:
    mark = '  ← 当前 m_blind' if abs(t - mb) < 1e-9 else ''
    print(f'             F({t:.2f}) = {v}{mark}')
vv = g('viol_blind')
mk = f'bgap_lt_{f"{mb:g}".replace(".", "p")}'
if vv is not None and g(mk) is not None:
    ok = '✓ 逐位相等' if abs(vv - g(mk)) < 1e-6 else '🔴 不相等 ⇒ 新读数写错了，CDF 不可信'
    print(f'    [自校验] viol_blind={vv} vs {mk}={g(mk)}  {ok}')
mono = all((a[1] or 0) <= (b[1] or 0) + 1e-9 for a, b in zip(cdf, cdf[1:]))
print(f'    [自校验] CDF 单调不减：{"✓" if mono else "🔴 违反"}')
task, bl = g('task'), g('blind')
if task is not None and bl is not None:
    share = lb * bl / (task + lb * bl + 1e-12)
    print(f'    [定标] task={task:.6f}  l_blind={bl:.6f}  λ_blind·l_blind 占 loss ≈ {share * 100:.2f}%')
    print(f'           （armC 的 λ_rank·l_rank 只占 0.22% ⇒ 数值上是死的；目标 ~5%，<1% 或 >25% 都要改 λ_blind）')
print(f'    [定标] 其余：c_blind={g("c_blind")} q_blind={g("q_blind")} d_blind={g("d_blind")} '
      f'd_blind_gated={g("d_blind_gated")} gate_frac_blind={g("gate_frac_blind")}')
print('    [定标] 🔴 判读规则（预先声明的两段式，改动只允许发生在这一个时点、改完换新 TAG）：')
print('           ① 优先找**平台**（相邻阈值之间 F 变化 <0.05）⇒ d_blind 是双峰的（先验题 d≈0 / 证据题 d 大），')
print('              把 m_blind 放进平台里 = 干净地分开两群，这是**理想**设定，不是退而求其次。')
print('           ② 没有平台就把 m_blind 内插到 F≈0.5；③ m_blind 必须留在 [0.05, 0.40]（>0.40 在 noul 上')
print('              近乎不可满足 ⇒ 无条件罚，与 RAC λ_corr=0.4 那档 acc 塌到 0.589 同类风险）。')
print('           ④ 若 F(0.05) 已经 >0.9 ⇒ 基座的信心压根不依赖证据在场，这本身就是**结论**，照实报，')
print('              不许为了把臂跑起来而硬抬 m_blind。')
PYEOF
    else
      log "  🔴 $OUT/measure/train_log.jsonl 不在 —— 20 步没跑完（八成是 OOM）；看 $OUT/logs/measure.log"
    fi
    log "  🔴 OOM 的处置：VIEWS=4 比 armD 多 1/3 行前向。**不要**降 VIEWS（会破坏与 armD 的 CE 行集同构）；"
    log "     先确认 OMP_NUM_THREADS=1 在位，再考虑 ACCUM 16（同样 700 optimizer step，峰值显存不变、墙钟翻倍）。"
  else
    log "  DRY: train ${ARM_ARGS[*]} ${RACD_ARGS[*]} --lr $LR --epochs 1 --limit $M_STEPS" \
        "--log-every 5 --init-from $SRC_CK --resume off --out $OUT/measure"
  fi
  exit 0
fi

if [ "$STAGE" = "train" ] || [ "$STAGE" = "both" ]; then
  log "=== 训练 $LIMIT 步 -> $OUT/racd1"
  if [ "$GO" = "1" ]; then
    _launch "${ARM_ARGS[@]}" "${RACD_ARGS[@]}" --lr "$LR" --epochs 1 --limit "$LIMIT" \
      --log-every "$LOG_EVERY" --save-every "$SAVE_EVERY" \
      --init-from "$SRC_CK" --resume auto \
      --out "$OUT/racd1" > "$OUT/logs/train.log" 2>&1
    tail -n 12 "$OUT/logs/train.log" | sed 's/^/    /'
    log "  -> $OUT/racd1（ckpt_last.pt / train_log.jsonl / slice_stats.json）"
    log "  看读数：tail $OUT/racd1/train_log.jsonl（每步一个 racd 块：d_tgt/d_dist/gate_frac/viol_*）"
  else
    log "  DRY: train ${ARM_ARGS[*]} ${RACD_ARGS[*]} --lr $LR --epochs 1 --limit $LIMIT" \
        "--log-every $LOG_EVERY --save-every $SAVE_EVERY --init-from $SRC_CK --resume auto" \
        "--out $OUT/racd1"
  fi
fi

if [ "$STAGE" = "eval" ] || [ "$STAGE" = "both" ]; then
  if [ "$GO" = "1" ]; then
    if [ ! -f "$OUT/racd1/ckpt_last.pt" ]; then
      log "FATAL: $OUT/racd1/ckpt_last.pt 不在 —— 先跑 STAGE=train"; exit 1
    fi
    if [ "$EVAL_LIMIT" != "0" ]; then
      log "  🔴 EVAL_LIMIT=$EVAL_LIMIT != 0：只覆盖前 $EVAL_LIMIT 题，不是报数口径"
    fi
    if [ "$EVAL_SHARDS" -gt 1 ]; then
      EVAL_LOG="$OUT/logs/eval_merge.log"
      log "=== 评估（$EVAL_SHARDS 片并行，卡 ${GPU_ARR[*]:0:EVAL_SHARDS}）：split=$EVAL_SPLIT bs=$EVAL_BS"
      log "  每片 = 连续行块；片产物 $OUT/racd1/verdicts_shard*_of_$EVAL_SHARDS.jsonl"
      pids=()
      for i in $(seq 0 $((EVAL_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES="${GPU_ARR[$i]}" "$PY" -u -m radev train eval \
          --run "$OUT/racd1" --split "$EVAL_SPLIT" --out "$OUT/racd1" \
          --limit "$EVAL_LIMIT" --bs "$EVAL_BS" \
          --slice-proc-lru "$SLICE_PROC_LRU" \
          --num-shards "$EVAL_SHARDS" --shard "$i" \
          > "$OUT/logs/eval_shard_$i.log" 2>&1 &
        pids+=("$!")
      done
      failed=()
      for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || failed+=("$i")
      done
      if [ "${#failed[@]}" -ne 0 ]; then
        log "FATAL: 片 ${failed[*]} 非零退出 —— 不 merge（缺片的 report 等于换个更小的 test）"
        for i in "${failed[@]}"; do tail -n 5 "$OUT/logs/eval_shard_$i.log" | sed "s/^/    [片$i] /"; done
        exit 1
      fi
      log "=== merge $EVAL_SHARDS 片 -> report（单卡 $(_eval_dev)，不推理，只并字典）"
      CUDA_VISIBLE_DEVICES="$(_eval_dev)" "$PY" -u -m radev train eval \
        --run "$OUT/racd1" --split "$EVAL_SPLIT" --out "$OUT/racd1" \
        --limit "$EVAL_LIMIT" --bs "$EVAL_BS" --merge-shards \
        > "$EVAL_LOG" 2>&1
      grep -h '行/秒' "$OUT"/logs/eval_shard_*.log "$EVAL_LOG" | sed 's/^/    /' || true
    else
      EVAL_LOG="$OUT/logs/eval.log"
      log "=== 评估（单卡 $(_eval_dev)）：split=$EVAL_SPLIT limit=$EVAL_LIMIT"
      CUDA_VISIBLE_DEVICES="$(_eval_dev)" "$PY" -u -m radev train eval \
        --run "$OUT/racd1" --split "$EVAL_SPLIT" --out "$OUT/racd1" --limit "$EVAL_LIMIT" \
        --bs "$EVAL_BS" --slice-proc-lru "$SLICE_PROC_LRU" > "$EVAL_LOG" 2>&1
    fi
    tail -n 10 "$EVAL_LOG" | sed 's/^/    /'
    log "  -> $OUT/racd1/report_$EVAL_SPLIT.md（对照 = $SRC/report_$EVAL_SPLIT.md）"
  else
    log "  DRY: eval --run $OUT/racd1 --split $EVAL_SPLIT --bs $EVAL_BS --slice-proc-lru $SLICE_PROC_LRU"
    if [ "$EVAL_SHARDS" -gt 1 ]; then
      log "       + $EVAL_SHARDS 片：--num-shards $EVAL_SHARDS --shard 0..$((EVAL_SHARDS - 1)) 一卡一片并行，"
      log "         全部退出后再跑一遍同样命令加 --merge-shards（这才出 report）"
    fi
  fi
fi

log "完成（STAGE=$STAGE GO=$GO NGPU=$NGPU OUT=$OUT LIMIT=$LIMIT VIEWS=$VIEWS LR=$LR BLIND=$BLIND λ_blind=$LAM_BLIND m_blind=$M_BLIND）"
