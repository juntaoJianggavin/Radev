#!/usr/bin/env bash
set -Eeuo pipefail
trap 'rc=$?; echo "[b] FATAL: 第 $LINENO 行 \`$BASH_COMMAND\` 非零退出（rc=$rc）" >&2; exit $rc' ERR

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python3}"
cd "$REPO" || { echo "[b] FATAL: 进不了 REPO=$REPO"; exit 1; }
export USABLE_GATE="${USABLE_GATE:-V2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SIZES="${SIZES:-0.8B}"
LLM_ROOT="${LLM_ROOT:-$REPO/pretrained}"
ENCODER="${ENCODER:-radar_ft}"
PURE2D="${PURE2D:-0}"
FEATS="${FEATS:-$([ "$PURE2D" = "1" ] && echo none || echo cache)}"
VISUAL_ROOT="${VISUAL_ROOT:-$REPO/feature_cache}"
SLICES="${SLICES:-32}"
SLICE_ROOT="${SLICE_ROOT:-$REPO/slice_cache}"
FREEZE_VISION="${FREEZE_VISION:-0}"
FUSE="${FUSE:-0}"
FUSION_POOL="${FUSION_POOL:-0}"
ORGAN2D="${ORGAN2D:-0}"

PROMPT_NO_COUNT="${PROMPT_NO_COUNT:-1}"

ORGAN_EVIDENCE="${ORGAN_EVIDENCE:-0}"
EVIDENCE_NOBOX="${EVIDENCE_NOBOX:-0}"
ORGAN_ROUTE="${ORGAN_ROUTE:-0}"
ORGAN_SLOTS="${ORGAN_SLOTS:-on}"
ORGAN_INDEX="${ORGAN_INDEX:-on}"
SLOT_PERM="${SLOT_PERM:-0}"
VIS_TOKENS="${VIS_TOKENS:-all}"
LOSS="${LOSS:-combined}"
ORGAN_CROP="${ORGAN_CROP:-off}"
CROP_MARGIN="${CROP_MARGIN:-0.1}"
CROP_CACHE_N="${CROP_CACHE_N:-24}"
SLICE_LRU="${SLICE_LRU:-8}"
TRAIN_SPLIT="${TRAIN_SPLIT:-train+val}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
RULES="${RULES:-explicit_negative,asserted_positive,hedged_positive,implied_negative_v2}"
OUT_ROOT="${OUT_ROOT:-$REPO/checkpoints}"
STAGE="${STAGE:-measure}"
RESUME="${RESUME:-0}"
ARM="${ARM:-align}"
LIMIT_STEPS="${LIMIT_STEPS:-1000}"
M_STEPS="${M_STEPS:-20}"
LIMIT_FULL="${LIMIT_FULL:-0}"
SAVE_EVERY="${SAVE_EVERY:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
BS="${BS:-1}"
ACCUM="${ACCUM:-8}"
LR="${LR:-2e-5}"
ALIGN_LR="${ALIGN_LR:-1e-3}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEV="${CUDA_DEV:-0}"
NGPU="${NGPU:-1}"
GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29500}"
GO="${GO:-0}"
TINY="${TINY:-0}"

log() { echo "[b] $*"; }

TRAIN_JSON="${TRAIN_JSON:-$REPO/data/iid/train_val.json}"
EVAL_JSON="${EVAL_JSON:-$REPO/data/iid/test.json}"
ORGAN_MAP="${ORGAN_MAP:-$REPO/organ_map.tsv}"
DATA_TAG="${DATA_TAG:-}"
if [ "$TRAIN_JSON" = "internal" ] || [ "$EVAL_JSON" = "internal" ] \
   || [ "$TRAIN_JSON" = "off" ] || [ "$EVAL_JSON" = "off" ]; then
  if [ "$TRAIN_JSON:$EVAL_JSON" != "internal:internal" ] \
     && [ "$TRAIN_JSON:$EVAL_JSON" != "off:off" ]; then
    log "FATAL: 数据面要成对：TRAIN_JSON=$TRAIN_JSON EVAL_JSON=$EVAL_JSON"
    exit 1
  fi
  TRAIN_JSON=""; EVAL_JSON=""; DATA_TAG="-tsv"
elif [ -n "$TRAIN_JSON" ] && [ -z "$DATA_TAG" ]; then
  case "$TRAIN_JSON" in
    *data/iid*) DATA_TAG="-iid" ;;
    *) DATA_TAG="-json" ;;
  esac
fi

if [ "$SLICES" -lt 1 ] && [ "$GO" = "1" ] && [ "$STAGE" != "measure" ]; then
  log "🔴 SLICES=0：这一臂不排任何图像插槽，原生通路整个关掉。它是**对照臂**（B 支涨的"
  log "   分数要能归因到切片，否则就是换了宿主），不是 B 支本身。产物目录带 -s0 后缀，"
  log "   别和 s$SLICES 那几臂混着读。"
fi
if [ "$SLICES" -ne 32 ] && [ "$SLICES" -gt 0 ] && [ "$GO" = "1" ] && [ "$STAGE" != "measure" ]; then
  log "⚠️ SLICES=$SLICES ≠ 32：这一臂是切片预算消融，题面里的「32 axial CT slices」"
  log "   会在装载时改写成无数量版本（训练与评估同一条路）；读数不与 s32 臂直接比"
  log "   （目录已按 -s$SLICES 分开）。"
fi
if [ "$RESUME" = "1" ]; then
  if [ "$STAGE" != "full" ]; then
    log "FATAL: RESUME=1 只能配 STAGE=full（收到 STAGE=$STAGE）；对齐段不重跑"
    exit 1
  fi
  if [ "${LIMIT_FULL:-0}" -le 0 ]; then
    log "FATAL: RESUME=1 需要 LIMIT_FULL=<该臂原定总步数>（0 会把整个 epoch 再跑一遍）"
    exit 1
  fi
fi
if [ "$PURE2D" = "1" ]; then
  if [ "$SLICES" -lt 1 ]; then
    log "FATAL: PURE2D=1 要 SLICES>0 —— 切片是纯 2D 唯一视觉入口"; exit 1
  fi
  if [ "$FEATS" != "none" ]; then
    log "FATAL: PURE2D=1 要 FEATS=none（收到 $FEATS）—— 3D 特征一条都不读"; exit 1
  fi
  if [ "$FUSE" = "1" ]; then
    log "FATAL: PURE2D=1 与 FUSE=1 不合 —— 融合的 K/V 来自 3D 特征"; exit 1
  fi
  if [ "$ORGAN_SLOTS" != "off" ]; then
    log "FATAL: PURE2D=1 要 ORGAN_SLOTS=off —— 纯 2D 序列里没有器官槽"; exit 1
  fi
  if [ "$ORGAN_CROP" != "off" ]; then
    log "FATAL: PURE2D=1 与 ORGAN_CROP=$ORGAN_CROP 不合 —— 裁剪臂读 3D 定位列"; exit 1
  fi
  if [ "$STAGE" = "align" ] || [ "$STAGE" = "both" ]; then
    log "FATAL: PURE2D=1 是直接全参微调（STAGE=full），没有对齐段；projector 已冻，"
    log "       --freeze-llm 的对齐段无事可训"; exit 1
  fi
  if [ "$STAGE" = "full" ] && [ "$RESUME" != "1" ] && [ "$ARM" != "scratch" ]; then
    log "FATAL: PURE2D=1 直接全参微调要 ARM=scratch（收到 ARM=$ARM）—— ARM=align 会去找"
    log "       不存在的对齐段 ckpt"; exit 1
  fi
fi
if [ "$FUSE" = "1" ] && [ "$SLICES" -lt 1 ]; then
  log "FATAL: FUSE=1 但 SLICES=0 —— 融合的 Q 流来自切片，原生通道关着融合无物可注"
  exit 1
fi
if [ "$ORGAN_CROP" != "off" ] && [ "$SLICES" -lt 1 ]; then
  log "FATAL: ORGAN_CROP=$ORGAN_CROP 但 SLICES=0 —— 裁剪臂无切片流（无物可裁）"
  exit 1
fi
if [ "$ORGAN_SLOTS" = "off" ] && [ "$FUSE" != "1" ] && [ "$PURE2D" != "1" ]; then
  log "FATAL: ORGAN_SLOTS=off 必须配 FUSE=1 —— 器官槽关掉后 3D 只以融合 K/V 身份进入，"
  log "       两个都不开 = 纯 2D 宿主基线（那是另一条臂，用 PURE2D=1），对齐段也会空转"
  exit 1
fi
if [ "$ORGAN_SLOTS" = "off" ] && [ "$SLICES" -lt 1 ]; then
  log "FATAL: ORGAN_SLOTS=off 但 SLICES=0 —— 融合的 Q 流（切片 token）不存在，融合无物可注"
  exit 1
fi
if [ "$FUSION_POOL" != "0" ]; then
  if [ "$FUSE" = "1" ]; then
    log "   池化臂=两跳（FUSE=1：切片先读器官，器官 query 再池化融合后的切片 token）"
  else
    log "   池化臂=单跳（FUSE=0：器官 query 直接池化原生塔输出，2D 侧无器官条件化）"
  fi
  if [ "$SLICES" -lt 1 ]; then
    log "FATAL: FUSION_POOL=$FUSION_POOL 但 SLICES=0 —— 池化的输入是切片 token"
    exit 1
  fi
  if [ "$ORGAN_SLOTS" != "on" ]; then
    log "FATAL: FUSION_POOL=$FUSION_POOL 要 ORGAN_SLOTS=on —— query 来自器官槽"
    exit 1
  fi
  if [ "$PURE2D" = "1" ]; then
    log "FATAL: FUSION_POOL 与 PURE2D=1 不合 —— 纯 2D 没有 3D 特征/融合/池化"
    exit 1
  fi
fi
if [ "$ORGAN2D" = "1" ]; then
  if [ "$FUSE" = "1" ]; then
    log "FATAL: ORGAN2D=1 与 FUSE=1 不合 —— organ2d 的序列里没有切片 token，融合无 Q 流"
    exit 1
  fi
  if [ "$FUSION_POOL" != "0" ]; then
    log "FATAL: ORGAN2D=1 与 FUSION_POOL=$FUSION_POOL 不合 —— 两者都改切片段的序列结构"
    exit 1
  fi
  if [ "$ORGAN_CROP" != "off" ]; then
    log "FATAL: ORGAN2D=1 与 ORGAN_CROP=$ORGAN_CROP 不合 —— 裁剪改切片坐标系，organ_box 归属会断"
    exit 1
  fi
  if [ "$PURE2D" = "1" ]; then
    log "FATAL: ORGAN2D=1 与 PURE2D=1 不合 —— 残差基座是 projector 器官槽，纯 2D 没有它"
    exit 1
  fi
  if [ "$SLICES" -lt 1 ]; then
    log "FATAL: ORGAN2D=1 要 SLICES>0 —— 2D 证据来自宿主塔对切片的前向"
    exit 1
  fi
  if [ "$ORGAN_SLOTS" != "on" ]; then
    log "FATAL: ORGAN2D=1 要 ORGAN_SLOTS=on —— 器官槽不在序列里残差就无处注入"
    exit 1
  fi
  if [ "$FEATS" = "zeros" ]; then
    log "FATAL: ORGAN2D=1 配 FEATS=zeros：没有 npz 就没有 organ_box，聚合恒无效"
    exit 1
  fi
fi

if [ "$EVIDENCE_NOBOX" = "1" ] && [ "$ORGAN_EVIDENCE" != "1" ]; then
  log "FATAL: EVIDENCE_NOBOX=1 要 ORGAN_EVIDENCE=1 —— 它放宽的是证据 token 的 K/V 取材，"
  log "       没有证据臂就没有可放宽的东西（无几何归属的器官池化是 FUSION_POOL，另一条臂）"
  exit 1
fi
if [ "$ORGAN_EVIDENCE" = "1" ]; then
  if [ "$FUSE" = "1" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 与 FUSE=1 不合 —— 两条臂都改器官段与切片段的耦合方式"
    exit 1
  fi
  if [ "$FUSION_POOL" != "0" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 与 FUSION_POOL=$FUSION_POOL 不合 —— 两者都改器官段的序列结构"
    exit 1
  fi
  if [ "$ORGAN2D" = "1" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 与 ORGAN2D=1 不合 —— 两条臂都改器官段的序列/嵌入结构"
    exit 1
  fi
  if [ "$ORGAN_CROP" = "bbox" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 与 ORGAN_CROP=bbox 不合 —— 平面裁剪改掉切片坐标系，"
    log "       organ_box 的区域归属就断（off/z/mask 保持全幅平面，可用）"
    exit 1
  fi
  if [ "$PURE2D" = "1" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 与 PURE2D=1 不合 —— 证据的 query 是 projector 器官嵌入，纯 2D 没有它"
    exit 1
  fi
  if [ "$SLICES" -lt 1 ]; then
    log "FATAL: ORGAN_EVIDENCE=1 要 SLICES>0 —— 证据的 K/V 来自宿主塔对切片的前向"
    exit 1
  fi
  if [ "$ORGAN_SLOTS" != "on" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 要 ORGAN_SLOTS=on —— 器官槽不在序列里证据 token 无槽可依"
    exit 1
  fi
  if [ "$FEATS" = "zeros" ]; then
    log "FATAL: ORGAN_EVIDENCE=1 配 FEATS=zeros：没有 npz 就没有 organ_box，区域归属恒空"
    exit 1
  fi
fi

case "$ORGAN_ROUTE" in
  0|1|visual|perorgan) ;;
  *) log "FATAL: ORGAN_ROUTE 只能是 0 / 1 / visual / perorgan（收到 $ORGAN_ROUTE）"; exit 2 ;;
esac
if [ "$ORGAN_ROUTE" = "1" ] || [ "$ORGAN_ROUTE" = "visual" ] \
   || [ "$ORGAN_ROUTE" = "perorgan" ]; then
  if [ "$ORGAN_SLOTS" != "on" ]; then
    log "FATAL: ORGAN_ROUTE=$ORGAN_ROUTE 要 ORGAN_SLOTS=on —— 门控缩放的是器官槽，序列里没有槽就无物可缩放"
    exit 1
  fi
fi

case "$ORGAN_INDEX" in
  on|off) ;;
  *) log "FATAL: ORGAN_INDEX 只能是 on / off（收到 $ORGAN_INDEX）"; exit 2 ;;
esac
case "$VIS_TOKENS" in
  all|organs|whole) ;;
  *) log "FATAL: VIS_TOKENS 只能是 all / organs（A4）/ whole（A3）（收到 $VIS_TOKENS）"; exit 2 ;;
esac
case "$SLOT_PERM" in
  ''|*[!0-9]*) log "FATAL: SLOT_PERM 要一个非负整数种子（收到 $SLOT_PERM）"; exit 2 ;;
esac
if { [ "$ORGAN_INDEX" = "off" ] && [ "$SLOT_PERM" = "0" ]; } \
   || { [ "$ORGAN_INDEX" = "on" ] && [ "$SLOT_PERM" != "0" ]; }; then
  log "FATAL: A1 只带了一半（ORGAN_INDEX=$ORGAN_INDEX SLOT_PERM=$SLOT_PERM）。"
  log "       撤身份嵌入而行序固定 ⇒「第 20 槽是肝」这件事仍由位置携带，什么也没撤；"
  log "       只打乱行序而留着身份嵌入 ⇒ 身份跟着位置走，那是另一条主张。"
  log "       两半都要：ORGAN_INDEX=off SLOT_PERM=<非 0 种子>（对照臂 = on + 0）。"
  exit 1
fi

"$PY" - <<'PYEOF' || { log "FATAL: $PY 缺 torch/monai —— 用 PY=/path 指到正确的 env"; exit 1; }
import torch, monai, numpy, transformers
print(f"[b] 解释器 ok：torch {torch.__version__} / monai {monai.__version__} / "
      f"transformers {transformers.__version__} / cuda_available={torch.cuda.is_available()}")
PYEOF

if [ -n "$TRAIN_JSON" ]; then
  for f in "$TRAIN_JSON" "$EVAL_JSON"; do
    [ -s "$f" ] || { log "FATAL: 数据面文件不在或为空：$f"; exit 1; }
  done
  [ "$TRAIN_JSON" != "$EVAL_JSON" ] || { log "FATAL: train/eval 指向同一个文件"; exit 1; }
  log "数据面=导出 JSON：train=$(basename "$TRAIN_JSON") eval=$(basename "$EVAL_JSON")"
  log "  train $TRAIN_JSON  $(stat -Lc '%y' "$TRAIN_JSON" | cut -d. -f1)"
  log "  eval  $EVAL_JSON  $(stat -Lc '%y' "$EVAL_JSON" | cut -d. -f1)"
  log "  训练=user 题面 + assistant 裸符号/Answer 段（combined 候选损失；ce 为 token CE）；评估=逐候选 NLL（score_exact）"
else
  log "数据面=内部题面：与 test 导出题面不是同一串 token，产物不许与 -json 臂混读"
fi

for s in $SIZES; do
  d="$LLM_ROOT/Qwen3.5-$s"
  [ -f "$d/config.json" ] || { log "FATAL: 缺 $d/config.json（Qwen3.5 梯子没下全：#105 的下载件）"; exit 1; }
  "$PY" - "$d" <<'PYEOF' || exit 1
import json, sys
cfg = json.load(open(sys.argv[1] + '/config.json'))
vc = cfg.get('vision_config')
if vc is None:
    print(f"[b] FATAL: {sys.argv[1]} 的 config 没有 vision_config —— 纯文本基座喂不了切片"); sys.exit(1)
mg = vc.get('merge_size') or vc.get('spatial_merge_size') or '?'
print(f"[b]   Qwen3.5 宿主就位：{sys.argv[1].split('/')[-1]}，"
      f"vision depth={vc.get('depth')} merge={mg} "
      f"image_token_id={cfg.get('image_token_id')}")
PYEOF
  n=$(find "$SLICE_ROOT" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
  log "  切片缓存 $SLICE_ROOT：现有 $n 个 npz（空不是错误，第一例现读现算并 write-through；"
  log "  但 epoch 首遍每卷都要过一遍 volume 解码 —— 这笔钱算进 measure 的读数里）"
done

if [ "$FEATS" = "cache" ]; then
  n=$(find "$VISUAL_ROOT" -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
  log "RADAR 通路 feats=cache：根 $VISUAL_ROOT 现有 $n 个 npz"
  if [ "$n" -lt 1 ] && [ "$TINY" != "1" ]; then
    log "FATAL: 这支编码器（$ENCODER）一个 npz 都没有。先跑 8 卡编码："
    log "       ENCODER=$ENCODER OUT=$VISUAL_ROOT GO=1 bash $REPO/build_features.sh"
    log "       （或 FEATS=live 明确选择在训练里现算；FEATS=zeros 只在 TINY=1 的自检里允许）"
    exit 1
  fi
elif [ "$FEATS" = "live" ]; then
  log "RADAR 通路 feats=live：write-through 根 $VISUAL_ROOT（现算 ≈10 s/例 GPU）"
elif [ "$FEATS" = "zeros" ] && [ "$TINY" = "1" ]; then
  log "FEATS=zeros 且 TINY=1：自检模式，指标一律不许外引"
elif [ "$FEATS" = "none" ] && [ "$PURE2D" = "1" ]; then
  log "RADAR 通路 feats=none：纯 2D 臂不读任何 3D 特征（无缓存、无前端）"
else
  log "FATAL: FEATS=$FEATS —— 只接受 cache / live（zeros 只在 TINY=1 自检里允许；"
  log "       none 只配 PURE2D=1）"; exit 1
fi

if [ "$DEVICE" != "cpu" ] && [ "$TINY" != "1" ] && [ "$GO" = "1" ]; then
  need=${FREE_MIN_MIB:-24576}
  if [ "$NGPU" -gt 1 ]; then
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    [ "${#GPU_ARR[@]}" -ge "$NGPU" ] || { log "FATAL: GPUS=$GPUS 只有 ${#GPU_ARR[@]} 张，不够 NGPU=$NGPU"; exit 1; }
    for gi in $(seq 0 $((NGPU - 1))); do
      g="${GPU_ARR[$gi]}"
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null | tr -d ' ')
      [ -n "${free:-}" ] || { log "FATAL: 看不到卡 $g"; exit 1; }
      log "  卡 $g 空闲 $((free / 1024)) GiB（门槛 $((need / 1024)) GiB）"
      if [ "$free" -lt "$need" ]; then
        log "FATAL: 卡 $g 空闲不够。当前占用者：" >&2
        nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' >&2 || true
        log "  不挤别人。换卡（GPUS=...）或等。" >&2
        exit 1
      fi
    done
  else
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_DEV" 2>/dev/null | tr -d ' ')
    [ -n "${free:-}" ] || { log "FATAL: 看不到卡 $CUDA_DEV"; exit 1; }
    log "  卡 $CUDA_DEV 空闲 $((free / 1024)) GiB（门槛 $((need / 1024)) GiB）"
    if [ "$free" -lt "$need" ]; then
      log "FATAL: 卡 $CUDA_DEV 空闲不够。当前占用者：" >&2
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | sed 's/^/  /' >&2 || true
      log "  不挤别人。换卡（CUDA_DEV=k）或等。" >&2
      exit 1
    fi
  fi
fi

FZ=()
if [ "$FREEZE_VISION" = "1" ]; then
  if [ "$SLICES" -lt 1 ]; then
    log "FREEZE_VISION=1 但 SLICES=0：原生塔不存在，这个开关没有对象 —— 不加 --freeze-vision"
  else
    FZ=(--freeze-vision)
  fi
fi
FU=()
if [ "$FUSE" = "1" ]; then
  FU=(--fuse)
fi
FP=()
if [ "$FUSION_POOL" != "0" ]; then
  FP=(--fusion-pool "$FUSION_POOL")
fi
O2D=()
if [ "$ORGAN2D" = "1" ]; then
  O2D=(--organ2d)
fi
OEV=()
if [ "$ORGAN_EVIDENCE" = "1" ]; then
  OEV=(--organ-evidence)
fi
EVNB=()
if [ "$EVIDENCE_NOBOX" = "1" ]; then
  EVNB=(--evidence-no-box)
fi
ORT=()
ORT_TAG="-ort$ORGAN_ROUTE"
case "$ORGAN_ROUTE" in
  0) ORT_TAG="-ort0" ;;
  1) ORT=(--organ-route)
     ORT_TAG="-ort1" ;;
  visual) ORT=(--organ-route --organ-route-visual)
          ORT_TAG="-ortv" ;;
  perorgan) ORT=(--organ-route --organ-route-perorgan)
            ORT_TAG="-ortp" ;;
esac
PNC=()
if [ "$PROMPT_NO_COUNT" = "1" ]; then
  PNC=(--prompt-no-count)
fi
OC=()
if [ "$ORGAN_CROP" != "off" ]; then
  OC=(--organ-crop "$ORGAN_CROP" --crop-margin "$CROP_MARGIN" --crop-cache-n "$CROP_CACHE_N")
fi
OS=()
if [ "$ORGAN_SLOTS" = "off" ]; then
  OS=(--organ-slots off)
fi
P2D=()
if [ "$PURE2D" = "1" ]; then
  P2D=(--pure2d)
fi
VPT=()
if [ "$VIS_TOKENS" != "all" ]; then
  VPT=(--vis-tokens "$VIS_TOKENS")
fi
IDX=()
SPM=()
if [ "$ORGAN_INDEX" = "off" ]; then
  IDX=(--organ-index off)
  SPM=(--slot-perm "$SLOT_PERM")
fi

common() {
  local c=(--feats "$FEATS" --encoder "$ENCODER" --train-split "$TRAIN_SPLIT"
           --rules "$RULES" --dtype bfloat16 --bs "$BS" --accum "$ACCUM" --seed 0 --loss "$LOSS"
           --slices "$SLICES" --slice-root "$SLICE_ROOT" --slice-lru "$SLICE_LRU"
           --llm "$LLM_ROOT/Qwen3.5-$1" "${FZ[@]}" "${FU[@]}" "${FP[@]}" "${O2D[@]}" "${OEV[@]}" "${EVNB[@]}" "${ORT[@]}" "${PNC[@]}" "${OC[@]}" "${OS[@]}" "${P2D[@]}" "${VPT[@]}" "${IDX[@]}" "${SPM[@]}")
  if [ "$FEATS" = "cache" ]; then c+=(--cache-root "$VISUAL_ROOT");
  elif [ "$FEATS" = "live" ]; then c+=(--visual-root "$VISUAL_ROOT"); fi
  if [ -n "$TRAIN_JSON" ]; then c+=(--train-json "$TRAIN_JSON" --eval-json "$EVAL_JSON"); fi
  if [ -n "${ORGAN_MAP:-}" ]; then c+=(--organ-map "$ORGAN_MAP"); fi
  if [ "$TINY" = "1" ]; then c+=(--tiny); fi
  if [ "$SAVE_EVERY" -gt 0 ]; then c+=(--save-every "$SAVE_EVERY"); fi
  printf '%s\n' "${c[@]}"
}

_launch_train() {
  if [ "$NGPU" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 "$PY" -m torch.distributed.run \
      --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
      -m radev train train "$@"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_DEV" PYTHONFAULTHANDLER=1 "$PY" -u -m radev train train "$@"
  fi
}
_eval_dev() {
  if [ "$NGPU" -gt 1 ]; then echo "${GPUS%%,*}"; else echo "$CUDA_DEV"; fi
}

run_one() {
  local s="$1" stage_out align_ckpt p2d_tag="" fp_tag="" o2d_tag="" oev_tag="" evnb_tag="" nc_tag="" vt_tag="" idx_tag="" spm_tag=""
  if [ "$PURE2D" = "1" ]; then p2d_tag="-p2d"; fi
  if [ "$FUSION_POOL" != "0" ]; then fp_tag="-fp$FUSION_POOL"; fi
  if [ "$ORGAN2D" = "1" ]; then o2d_tag="-o2d"; fi
  if [ "$ORGAN_EVIDENCE" = "1" ]; then oev_tag="-oev"; fi
  if [ "$EVIDENCE_NOBOX" = "1" ]; then evnb_tag="-evnb"; fi
  if [ "$PROMPT_NO_COUNT" = "1" ]; then nc_tag="-nc"; fi
  if [ "$VIS_TOKENS" != "all" ]; then vt_tag="-vt$VIS_TOKENS"; fi
  if [ "$ORGAN_INDEX" = "off" ]; then idx_tag="-idxoff"; spm_tag="-spm$SLOT_PERM"; fi
  stage_out="$OUT_ROOT/$s-$ARM-fz$FREEZE_VISION-fs$FUSE-os$ORGAN_SLOTS-oc$ORGAN_CROP-s$SLICES-l$LOSS$DATA_TAG$p2d_tag$fp_tag$o2d_tag$oev_tag$evnb_tag$nc_tag$ORT_TAG$vt_tag$idx_tag$spm_tag"
  mkdir -p "$stage_out/logs"
  align_ckpt="$stage_out/align/ckpt_last.pt"
  mapfile -t common_args < <(common "$s")

  if [ "$STAGE" = "measure" ]; then
    log "=== 计时：B 支 $M_STEPS 步（bs $BS，slices $SLICES，ngpu $NGPU），不产成绩，只看墙钟"
    if [ "$GO" = "1" ]; then
      local t0=$(date +%s)
      _launch_train "${common_args[@]}" --lr "$LR" --epochs 1 --limit "$M_STEPS" \
        --out "$stage_out/measure" > "$stage_out/logs/measure.log" 2>&1
      local el=$(( $(date +%s) - t0 ))
      tail -n 4 "$stage_out/logs/measure.log" | sed 's/^/    /'
      awk -v e="$el" -v k="$M_STEPS" -v b="$BS" -v sl="$SLICES" -v ng="$NGPU" 'BEGIN {
        printf "    [b] => %.2f s/step（%d 卡 DDP，全局 bs=%d）；一个 epoch 的 22.3e4 题 ≈ %.1f h\n",
               e/k, ng, b*ng, e/k*223286/(b*ng)/3600
        printf "    [b]    其中每题 %s 片过宿主塔（按题付，不按卷付）\n", sl }'
      log "  拿着这个数再定 LIMIT_STEPS / SLICES，然后 STAGE=both"
    else
      log "  DRY: train ${common_args[*]} --limit $M_STEPS --out $stage_out/measure"
    fi
    return
  fi

  if [ "$STAGE" = "align" ] || [ "$STAGE" = "both" ]; then
    local what="projector"
    if [ "$ORGAN_SLOTS" = "off" ]; then
      what="fusion（organ-slots off：序列无器官槽，projector 不进计算图）"
    elif [ ${#FU[@]} -gt 0 ]; then
      what="projector+fusion"
      if [ "$FUSION_POOL" != "0" ]; then what="projector+fusion+pool"; fi
    elif [ "$FUSION_POOL" != "0" ]; then
      what="projector+pool"
    fi
    log "=== Qwen3.5-$s 对齐段（${NGPU}卡）：冻结 LLM（含原生塔），只训 $what（$LIMIT_STEPS 步，lr=$ALIGN_LR）"
    if [ "$GO" = "1" ]; then
      _launch_train "${common_args[@]}" --lr "$ALIGN_LR" --freeze-llm --epochs 1 --limit "$LIMIT_STEPS" \
        --out "$stage_out/align" > "$stage_out/logs/align.log" 2>&1
      tail -n 6 "$stage_out/logs/align.log" | sed 's/^/    /'
      log "  -> $stage_out/align（对齐段指标不许当 ③ 报数）"
    else
      log "  DRY: train ${common_args[*]} --lr $ALIGN_LR --freeze-llm --limit $LIMIT_STEPS"
    fi
  fi

  if [ "$STAGE" = "full" ] || [ "$STAGE" = "both" ]; then
    local init=() resume=()
    if [ "$RESUME" = "1" ]; then
      local full_ck="$stage_out/full/ckpt_last.pt"
      if [ ! -f "$full_ck" ]; then
        log "FATAL: RESUME=1 但 $full_ck 不在 —— 没有可接续的全参段" >&2
        exit 1
      fi
      resume=(--resume "$full_ck")
      local tl="$stage_out/full/train_log.jsonl"
      if [ -f "$tl" ] && [ "${FORCE_RESUME:-0}" != "1" ]; then
        local age=$(( $(date +%s) - $(stat -c %Y "$tl") ))
        if [ "$age" -lt 7200 ]; then
          log "FATAL: $tl 在 ${age}s 前还在写 —— 旧 job 可能仍在跑；确认已死再续，或 FORCE_RESUME=1" >&2
          exit 1
        fi
      fi
      if [ -f "$stage_out/full/run_info.json" ] \
         && [ ! -f "$stage_out/full/run_info.pre_resume.json" ]; then
        cp -p "$stage_out/full/run_info.json" "$stage_out/full/run_info.pre_resume.json"
      fi
      log "=== 续跑：$full_ck（LIMIT_FULL=$LIMIT_FULL 为原定总步数）"
    elif [ "$ARM" = "align" ]; then
      if [ "$GO" = "1" ] && [ ! -f "$align_ckpt" ]; then
        log "FATAL: ARM=align 要对齐段产物 $align_ckpt，但它不在 —— 先跑 STAGE=align" >&2
        exit 1
      fi
      init=(--init-from "$align_ckpt")
    fi
    FZ_EFF=off
    if [ ${#FZ[@]} -gt 0 ]; then FZ_EFF=on; fi
    FU_EFF=off
    if [ ${#FU[@]} -gt 0 ]; then FU_EFF=on; fi
    log "=== Qwen3.5-$s 全参段（${NGPU}卡）：ARM=$ARM slices=$SLICES freeze_vision=$FZ_EFF fuse=$FU_EFF fusion_pool=$FUSION_POOL organ_slots=$ORGAN_SLOTS 步数=${LIMIT_FULL} lr=$LR"
    if [ "$GO" = "1" ]; then
      _launch_train "${common_args[@]}" "${init[@]}" "${resume[@]}" --lr "$LR" \
        --epochs 1 --limit "$LIMIT_FULL" \
        --out "$stage_out/full" > "$stage_out/logs/full.log" 2>&1
      tail -n 6 "$stage_out/logs/full.log" | sed 's/^/    /'
      log "=== 评估（单卡 $(_eval_dev)）：split=$EVAL_SPLIT limit=$EVAL_LIMIT"
      if [ "$EVAL_LIMIT" != "0" ]; then
        log "  🔴 EVAL_LIMIT=$EVAL_LIMIT != 0：这份 report 只覆盖前 $EVAL_LIMIT 题，"
        log "     不是报数口径（题不独立，按序取前 N 会偏向某几个 study）。只当流水线自检看。"
      fi
      CUDA_VISIBLE_DEVICES="$(_eval_dev)" "$PY" -u -m radev train eval \
        --run "$stage_out/full" --split "$EVAL_SPLIT" --out "$stage_out/full" \
        --limit "$EVAL_LIMIT" \
        > "$stage_out/logs/eval.log" 2>&1
      tail -n 8 "$stage_out/logs/eval.log" | sed 's/^/    /'
      log "  -> $stage_out/full/report_$EVAL_SPLIT.md"
    else
      log "  DRY: train ${common_args[*]} ${init[*]+--init-from $align_ckpt} ${resume[*]+--resume $stage_out/full/ckpt_last.pt} --lr $LR --limit $LIMIT_FULL"
      log "  DRY: eval --run $stage_out/full --split $EVAL_SPLIT"
    fi
  fi
}

for s in $SIZES; do run_one "$s"; done
[ "$GO" = "1" ] || log "DRY 结束：一个 GPU 都没碰。要真跑：GO=1 NGPU=4 GPUS=0,1,2,3 STAGE=measure FEATS=$FEATS PURE2D=$PURE2D SLICES=$SLICES FUSE=$FUSE FUSION_POOL=$FUSION_POOL ORGAN_SLOTS=$ORGAN_SLOTS ORGAN_CROP=$ORGAN_CROP LOSS=$LOSS OUT_ROOT=$OUT_ROOT bash $0"
log "完成（SIZES=\"$SIZES\" NGPU=$NGPU GPUS=$GPUS STAGE=$STAGE ARM=$ARM FEATS=$FEATS PURE2D=$PURE2D SLICES=$SLICES FREEZE_VISION=$FREEZE_VISION FUSE=$FUSE FUSION_POOL=$FUSION_POOL ORGAN_SLOTS=$ORGAN_SLOTS ORGAN_CROP=$ORGAN_CROP ORGAN_EVIDENCE=$ORGAN_EVIDENCE EVIDENCE_NOBOX=$EVIDENCE_NOBOX LOSS=$LOSS DATA=${DATA_TAG#-}）"
