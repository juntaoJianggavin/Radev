# Radev — decision-native multimodal CT interpretation

Radev reads a 3D abdominal CT volume plus a question with a small candidate set and
emits a probability distribution over the candidates in a single forward pass (no
autoregressive decoding). Training has three stages; inference is a restricted-softmax
readout over single-token candidates. The benchmark is **CT-Verdict** (IID + OOD).
This release contains only the published model arm.

## Layout

```
radev/                model + training + evaluation engine (package)
analysis/verdict_metrics.py   calibration metrics (ECE / Brier / KRPS) over verdicts
data/iid/             CT-Verdict-IID: train_val.json, test.json   (symbol answers; from HF, not in git)
data/ood/             CT-Verdict-OOD: test.json                   (prediction-only; from HF, not in git)
organ_map.tsv         question -> organ binding (text-only taxonomy)
question_keep.tsv     per-question split + rule_class list (factory families)
newq_keep.tsv         per-question split + family list (extended family)
geometry_scan.json    per-study organ geometry for the feature-cache planner
doc/                  dataset documentation, statistics, identifier/image remap tables
run_alignment.sh      stage 1: projector--evidence alignment
run_vft.sh            stage 2: verdict fine-tuning (VFT)
run_cct.sh            stage 3: counterfactual-calibrated training (CCT)
build_features.sh     frozen-encoder organ-feature cache builder
evaluate.sh           inference / evaluation
supervised_base.sh    shared engine launcher used by run_alignment.sh / run_vft.sh
```

The benchmark JSON files are **not stored in this repository** (size); download them
from the HF dataset `Gavinnnnnnnn/CT-Verdict` (`data/iid/{train_val,test}.json`,
`data/ood/test.json`) into the same paths under `data/`.
`data/iid` and `data/ood` contain only JSON. Image paths in the JSON are **relative**
(`iid/<sid>.nii.gz`, `ood/ood_<n>.nii.gz`); they resolve against `RADEV_IMAGE_ROOT`
(default = release root), so place volumes at `<root>/iid/` and `<root>/ood/`.
Images and model weights are not distributed (see External assets).

## Stages and commands

The engine is invoked as `python -m radev train {train,eval,floors}` and
`python -m radev radar-cache {plan,run}`. Launchers wrap these with the canonical knobs;
every path/parallelism knob is env-overridable (`REPO`, `PY`, `SRC`, `VISUAL_ROOT`,
`SLICE_ROOT`, `ORGAN_MAP`, `LLM_ROOT`, `ENCODER`, `NGPU`, `GPUS`, `ACCUM`, ...).
`GO=0` (default) is a dry run: prints the resolved command, touches no GPU; `GO=1` runs.
Training checkpoints are written under `checkpoints/` (`OUT_ROOT`); CCT initializes from
the VFT run dir (`SRC`, default `checkpoints/vft`) — set `SRC=<VFT_run_dir>`.

1. Organ-feature cache (frozen 3D encoder, per study):
   `GO=1 bash build_features.sh`
2. Alignment (projector + evidence attention only, host frozen):
   `GO=1 bash run_alignment.sh`
   = `--freeze-llm --loss combined --limit 1000 --lr 1e-3`
3. Verdict fine-tuning (VFT, full parameters):
   `GO=1 bash run_vft.sh`
   = `--loss combined --init-from <alignment>/ckpt_last.pt`
4. Counterfactual-calibrated training (CCT):
   `GO=1 SRC=<VFT_run_dir> bash run_cct.sh`
   = `--loss racd --racd-views 4 --racd-blind --racd-mask-whole --racd-lam-rank 0
   --racd-lam-null 0 --racd-lam-blind 0.2 --racd-m-blind 0.20 --init-from <VFT ckpt>`
5. Inference / evaluation:
   `RUN=<ckpt_dir> bash evaluate.sh`                              # IID test
   `RUN=<ckpt_dir> EVAL_JSON=$PWD/data/ood/test.json bash evaluate.sh`   # OOD test
   Writes per-question verdicts (`verdicts_*.jsonl`) and `report_test.md`
   (Acc / mBAcc / mAUC / MAE / QWK). Sharded eval: `--num-shards N --shard i` then
   `--merge-shards`.
6. Calibration metrics (ECE / Brier / KRPS) over the verdicts:
   `python analysis/verdict_metrics.py --run <ckpt_dir>`

## External assets (NOT included; obtain separately)

Override the launcher env vars to point at your copies.

- Language hosts: `pretrained/Qwen3.5-{0.8B,2B,4B,9B}` (`LLM_ROOT`).
- Frozen 3D encoder (RADAR+) checkpoint + text embedding, under `RADAR_CKPT_ROOT`
  (default `radar_ckpt/`): `checkpoint_radar_plus_finetuned.pth`,
  `infer_text_embedding.pt` (override with `--visual-ckpt` / `--ckpt` / `--text-emb`).
- Organ-feature cache: `feature_cache/` (`VISUAL_ROOT`; build with `build_features.sh`).
- Slice cache: `slice_cache/` (`SLICE_ROOT`).
- CT volumes (relative `image` paths), under `RADEV_IMAGE_ROOT` (default release root):
  `<root>/iid/<sid>.nii.gz` (25,270 studies) and `<root>/ood/ood_<n>.nii.gz` (421 volumes).
  See `doc/identifier_remap.md` and `doc/ood_volume_map.tsv`.

The encoder identity is `radar_ft`; feature caches built under any older identity must
be rebuilt with `build_features.sh`.

## Data format

- `data/iid/train_val.json` — 223,286 rows (train+val), SFT shape:
  `{id, image, messages:[user("<image>\n"+prompt), assistant(symbol)], answer_format:"symbol"}`;
  choice rows also carry `symbol_keys`. Per-row split/rule_class come from
  `question_keep.tsv` / `newq_keep.tsv` (required for train+val loading).
- `data/iid/test.json` — 56,825 rows, QA shape:
  `{id, image, question, answer, type, answer_format:"symbol"}` (+`symbol_keys` for choice).
- `data/ood/test.json` — 2,807 rows, same QA shape (prediction-only; no OOD label enters
  fitting and no OOD readout drives model selection).
- Question types: `noul` (binary finding presence), `choice` (multiple choice),
  `score` (ordinal severity, levels 0-4 rendered 1-5).
- `organ_map.tsv` — 241 question templates -> organ slots + group; OOD `ood_*` keys are
  intentionally unbound (whole-image fallback).
- See `doc/DATASET_DOC.md` for the labeling/audit pipeline and `doc/benchmark_section.tex`
  for the benchmark description.

## Note on this release

Only the published arm (alignment / VFT / CCT / inference + feature-cache builder) is
included; the dataset-construction subsystem is not part of this release. All `#`
comments were removed (docstrings preserved). Identifiers, image paths, the package name
(`radev`), and the encoder identity (`radar_ft`) are neutral; see `doc/identifier_remap.md`.
Shell launchers pass `bash -n`.
