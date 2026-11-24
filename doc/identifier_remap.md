# Identifier and image-path remapping (portable release)

The release JSON uses **relative image paths** and **neutral identifiers**. Images
are not distributed here; resolve every `image` field against your image root
(`RADEV_IMAGE_ROOT`, default = release root).

## Image paths

| split | release `image` | resolved location |
|---|---|---|
| IID | `iid/<sid>.nii.gz` | `<image_root>/iid/<sid>.nii.gz` |
| OOD | `ood/ood_<n>.nii.gz` | `<image_root>/ood/ood_<n>.nii.gz` |

Place IID volumes under `<image_root>/iid/` and OOD volumes under `<image_root>/ood/`.
The OOD source `imagesTr`/`imagesVa` split is flattened into `ood/` (the 421 volume
basenames are unique). `ood_volume_map.tsv` maps every OOD volume's original study id
and absolute source path to its release form.

## Identifiers

- IID ids are unchanged: `<sid>::<question_key>` (e.g. `AC4213645::finding_ascites`).
- OOD ids: the source dataset prefix was replaced by `ood_`, giving
  `ood_<n>::ood_<question_key>`. The prefix is replaced (not stripped to a bare
  `finding_` key) on purpose: `organ_map.tsv` binds bare `finding_*` keys to IID organ
  regions, while OOD questions are intentionally **unbound** (whole-image fallback, as
  in the paper). Keeping the `ood_` prefix preserves that unbound status, so model
  predictions are unchanged.
- Question text, answers, types, and `symbol_keys` are unchanged.

## Annotation artifacts (top level)

- `question_keep.tsv` / `newq_keep.tsv` — per-question `split` and `rule_class` (factory)
  / `family` (extended) lists, covering all 280,111 IID items. Training reads these to
  assign each row's split and to filter `finding_*` rows by `--rules`; they are required
  for `train_val.json` (loaded as `train+val`).
- `geometry_scan.json` — per-study organ geometry consumed by the feature-cache planner
  (`build_features.sh`); keyed by IID study id (unchanged).
- `organ_map.tsv` — question -> organ binding (text-only taxonomy).

## External derived assets

The frozen-encoder feature cache and slice cache are keyed by study id and encoder
identity. IID study ids are unchanged. The encoder identity is now `radar_ft` (formerly
a source-branded name); caches built under the old identity must be rebuilt with
`build_features.sh`. OOD study ids changed to `ood_<n>`; rebuild the OOD cache from the
release JSON or rename entries per `ood_volume_map.tsv`. The RADAR checkpoint and text
embedding are external assets expected at `RADAR_CKPT_ROOT` (default `radar_ckpt/`) as
`checkpoint_radar_plus_finetuned.pth` / `infer_text_embedding.pt`, overridable via
`--visual-ckpt` / `--ckpt` / `--text-emb`.
