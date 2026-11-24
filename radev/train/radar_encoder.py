"""RADAR 视觉塔的移植层 —— 冻结编码器，产出每器官 256-d 向量。

移植自 `/tmp/damo-radar@main` 的 RADAR_inference（Apache-2.0 / MONAI 许可头见
`vendor/dynamic_network_architectures/`）：

    VisionBranch（3D nnU-Net 编码器 + 三尺度 1x1x1 投影）: vision_branch.py（整套 vendor）
    query_tokens + MultiheadAttention(256,4) + per-organ vision_projs
        : inference_source_testset.py:296-306, 355-383
    滑窗 + 拼接 stitched_mask + 触边器官跳过 + center_crop 兜底
        : inference_source_testset.py:401-561
    logits = image_feat @ text_feat.T / temp; probs = softmax(-1)[...,1]
        : inference_source_testset.py:393-396

三条纪律：
* state_dict 的键名与 `ckpt['model']` 的视觉部分**完全同名**
  （visual_encoder.* / attention.* / vision_projs.* / query_tokens / temp，RADAR+ 另加
  `*_whole`），`load_state_dict(..., strict=True)` 能直接过。strict=False 是被禁止的 ——
  静默少加载就会静默对不上。
* **必须声明编码器身份**（`ENCODER_IDENTITIES`，a17）：核对 basename + 全文件 sha256 +
  视觉张量数。两支的实质差别是数据，不是形状：`radar_ft` 的视觉塔在 the source CT training cohort
  （≈ 我们的 train+val）上被更新过 ⇒ 它的逐例概率不是第三方佐证、② 级地板变成
  「在训练集上评估」，且发布 CSV 的逐位对拍锚点只属于 `pretrain`（C.43 ②③ 实测：
  视觉侧 466/492 个张量被改动，BatchNorm 统计 112/112 全变）。
  同一份权重还可以有第二种**推理形态**（`infer_form`，A2）—— 它改的是「怎么喂」，
  不改「喂什么权重」，所以那三件事对它们完全给不出区分，只能靠身份表里那一列。
* 权重哈希与加载结果记录在 `from_pretrained()` 的返回里，调用方负责写进 run info。
"""
from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data.utils import dense_patch_slices

from . import vendor as _vendor
from dynamic_network_architectures.vision_branch import VisionBranch

ORGANS = [
    '肾上腺', '主动脉', '竖脊肌', '脑', '锁骨', '大肠', '十二指肠', '食管', '面部', '股骨',
    '胆囊', "臀肌", '心脏', '髋关节', '肱骨', '髂动脉', '髂静脉', '髂腰肌', '下腔静脉', '肾',
    '肝', '肺', '胰腺', '门静脉', '肺动脉', '肋骨', '骶骨', '肩胛骨', '小肠', '脾',
    '胃', '气管', '膀胱', '颈椎', '腰椎', '胸椎',
]
N_ORGANS = len(ORGANS)
GALLBLADDER_ID = ORGANS.index('胆囊') + 1

ROI_SIZE = (96, 256, 384)
OVERLAP = 0.25
SW_BATCH = 1

TEST_ITEMS = [
    '主动脉_主动脉瘤', '主动脉_粥样硬化', '大肠_粘膜下水肿', '大肠_阑尾炎', '小肠_梗阻',
    '心脏_主动脉瓣钙化', '心脏_心影（脏）增大', '肝_肝内胆管扩张', '肝_肝大', '肝_脂肪肝',
    '肺_胸腔积液', '肺_膨胀不全', '肾_低密度影', '肾_囊肿', '肾_肾积水', '胆囊_结石',
    '胰腺_萎缩', '脾_脾大', '腰椎_骨折', '食管_裂孔疝', '胆囊_术后胆囊缺失',
]
SAG_ITEM = '胆囊_术后胆囊缺失'
PROB_ITEMS = [it for it in TEST_ITEMS if it != SAG_ITEM]

ENCODER_IDENTITIES: dict[str, dict[str, Any]] = {
    'pretrain': {
        'cls': 'RadarVisual',
        'basename': 'checkpoint_radar_pretrain.pth',
        'sha256': '5e8b1b50b92162fcafcf8231e51ffbe9d1be2be52ccff4f967e4beb71e4499f4',
        'key_prefixes': ('visual_encoder', 'attention', 'vision_projs', 'query_tokens', 'temp'),
        'n_tensors': 492,
        'text_lang': 'zh',
        'has_whole': False,
        'saw_radar_train': False,
        'published_csv_anchor': True,
        'infer_form': 'window',
    },
    'radar_ft': {
        'cls': 'RadarPlusVisual',
        'basename': 'checkpoint_radar_plus_finetuned.pth',
        'sha256': 'cd5aae5635de3712312c067dd82114cd4c215e7dcfc11eb3619dab73fcceaebc',
        'key_prefixes': ('visual_encoder', 'attention', 'vision_projs', 'query_tokens', 'temp',
                         'attention_whole', 'vision_projs_whole', 'query_tokens_whole'),
        'n_tensors': 499,
        'text_lang': 'en',
        'has_whole': True,
        'saw_radar_train': True,
        'published_csv_anchor': False,
        'infer_form': 'window',
    },
    'radar_ft_nowin': {
        'cls': 'RadarPlusVisual',
        'basename': 'checkpoint_radar_plus_finetuned.pth',
        'sha256': 'cd5aae5635de3712312c067dd82114cd4c215e7dcfc11eb3619dab73fcceaebc',
        'key_prefixes': ('visual_encoder', 'attention', 'vision_projs', 'query_tokens', 'temp',
                         'attention_whole', 'vision_projs_whole', 'query_tokens_whole'),
        'n_tensors': 499,
        'text_lang': 'en',
        'has_whole': True,
        'saw_radar_train': True,
        'published_csv_anchor': False,
        'infer_form': 'nowin',
    },
}

INFER_FORMS = ('window', 'nowin')
DEFAULT_INFER_FORM = 'window'


def infer_form_of(identity: str) -> str:
    """身份 -> 推理形态（缺 `infer_form` 的登记项按上游官方口径）。"""
    return ENCODER_IDENTITIES[identity].get('infer_form', DEFAULT_INFER_FORM)

VISUAL_KEY_PREFIXES = ENCODER_IDENTITIES['pretrain']['key_prefixes']


def sha256_of(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class RadarVisual(nn.Module):
    """键名与 ckpt['model'] 视觉部分逐字相同，strict 可加载。"""

    IDENTITY = 'pretrain'

    def __init__(self) -> None:
        super().__init__()
        self.visual_encoder = VisionBranch()
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.1, batch_first=True)
        self.vision_projs = nn.ModuleList([nn.Linear(256, 256) for _ in range(N_ORGANS)])
        self.query_tokens = nn.Parameter(torch.zeros(N_ORGANS, 256))
        self.temp = nn.Parameter(0.07 * torch.ones([]))

    @classmethod
    def from_pretrained(cls, ckpt_path: str, identity: str | None = None,
                        device: str = 'cpu') -> tuple['RadarVisual', dict[str, Any]]:
        """a17：`identity` 必填（除非路径恰好唯一命中某个已登记身份），核对三件事才放行。"""
        ident_key = identity
        if ident_key is None:
            hits = [k for k, v in ENCODER_IDENTITIES.items()
                    if v['basename'] == os.path.basename(ckpt_path)]
            if len(hits) != 1:
                raise ValueError(f'必须声明 --encoder：{os.path.basename(ckpt_path)} '
                                 f'命中 {len(hits)} 个登记身份（a17）')
            ident_key = hits[0]
        if ident_key not in ENCODER_IDENTITIES:
            raise ValueError(f'未登记的编码器身份 {ident_key}；可选 '
                             f'{sorted(ENCODER_IDENTITIES)}')
        ident = ENCODER_IDENTITIES[ident_key]
        if cls.__name__ != ident['cls']:
            raise ValueError(f'身份 {ident_key} 要用 {ident["cls"]}，拿到 {cls.__name__}（a17）')
        base = os.path.basename(ckpt_path)
        if base != ident['basename']:
            raise ValueError(f'编码器身份断言失败：声明 {ident_key} 应是 '
                             f'{ident["basename"]}，拿到 {base}（a17）')
        sha = sha256_of(ckpt_path)
        if sha != ident['sha256']:
            raise ValueError(f'编码器哈希断言失败：{base} 的 sha256={sha} 与登记的 '
                             f'{ident["sha256"]} 不符 —— 不是同一份权重（a17）')
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        sd = ck['model']
        keep = {k: v for k, v in sd.items() if k.split('.')[0] in ident['key_prefixes']}
        if len(keep) != ident['n_tensors']:
            raise ValueError(f'身份 {ident_key} 期望 {ident["n_tensors"]} 个视觉张量，'
                             f'实取 {len(keep)}（a17）')
        model = cls()
        missing, unexpected = model.load_state_dict(keep, strict=True)
        if missing or unexpected:
            raise RuntimeError(f'strict 加载不该有 missing/unexpected：{missing} / {unexpected}')
        model.eval().to(device)
        for p in model.parameters():
            p.requires_grad_(False)
        info = {
            'encoder_identity': ident_key,
            'ckpt_path': os.path.abspath(ckpt_path),
            'ckpt_sha256': sha,
            'n_tensors_visual': len(keep),
            'text_lang': ident['text_lang'],
            'has_whole': ident['has_whole'],
            'saw_radar_train': ident['saw_radar_train'],
            'published_csv_anchor': ident['published_csv_anchor'],
            'infer_form': ident.get('infer_form', DEFAULT_INFER_FORM),
        }
        return model, info


class RadarPlusVisual(RadarVisual):
    """RADAR+ 的两个分支都在，整图分支 = 一个 token 池化全部 patch（上游无滑窗那一路）。

    多出来的 3 个模块只在 radar_ft 里有权重（`checkpoint_radar_pretrain.pth` 的
    `*_whole` 是 0 个张量，C.43 ② 实测），所以 pretrain 身份不许用本类加载 —— strict
    加载会当场 missing，那正是我们想要的失败方式：静默用随机初始化的整图分支，
    下游拿到的是一列噪声而没有任何提示。
    """

    IDENTITY = 'radar_ft'

    def __init__(self) -> None:
        super().__init__()
        self.query_tokens_whole = nn.Parameter(torch.zeros(1, 256))
        self.attention_whole = nn.MultiheadAttention(embed_dim=256, num_heads=4,
                                                     dropout=0.1, batch_first=True)
        self.vision_projs_whole = nn.Linear(256, 256)


def _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap):
    if len(image_size) != num_spatial_dims:
        raise ValueError('image coord different from spatial dims.')
    if len(roi_size) != num_spatial_dims:
        raise ValueError('roi coord different from spatial dims.')
    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)


def _masks_to_boxes_3d(masks):
    """照抄上游 :29-59（[N,D,H,W] -> [N,6] x_min,y_min,z_min,x_max,y_max,z_max）。"""
    if masks.numel() == 0:
        return torch.zeros((0, 6), device=masks.device)
    d, h, w = masks.shape[-3:]
    z = torch.arange(0, d, dtype=torch.float, device=masks.device)
    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    z, y, x = torch.meshgrid(z, y, x, indexing='ij')
    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1).values
    x_min = x_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1).values
    y_min = y_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    z_mask = (masks * z.unsqueeze(0))
    z_max = z_mask.flatten(1).max(-1).values
    z_min = z_mask.masked_fill(~masks.bool(), float('inf')).flatten(1).min(-1).values
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=1)


PERCENTILE_Q = 0.5


def organ_boxes(mask: np.ndarray, pct: float = PERCENTILE_Q) -> np.ndarray:
    """[D,W,H] 分割（0=背景，1..36 器官 id）-> int16[36,6] 分位点 bbox，上下界均含。

    坐标序 (z0,z1,y0,y1,x0,x1)（z=dim0、y=dim1、x=dim2），与 `stitched_mask`、切片缓存
    同一坐标系（volume.load_and_prepare 的 pad 之后，无二次换算）。该类不在 mask 里 →
    该行全 -1；做并集时 -1 行被剔除（slice_frontend.union_crop）。
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f'organ_boxes 要 [D,W,H] 的分割，收到 {m.shape}')
    out = np.full((N_ORGANS, 6), -1, dtype=np.int16)
    for oi in range(N_ORGANS):
        z, y, x = np.nonzero(m == oi + 1)
        if z.size == 0:
            continue
        qs = np.percentile(np.stack([z, y, x], axis=1), [pct, 100.0 - pct], axis=0)
        lo, hi = np.floor(qs[0]).astype(np.int16), np.ceil(qs[1]).astype(np.int16)
        out[oi] = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    return out


def organ_zprofiles(mask: np.ndarray) -> np.ndarray:
    """[D,W,H] 分割（0=背景，1..36 器官 id）-> int32[36,D] 逐层体素计数。

    坐标与 `organ_boxes`/`stitched_mask` 同一坐标系（`volume.load_and_prepare` 的 pad
    之后，无二次换算）。逐层 bincount 而不是一次 ravel：96×256×384 的 int64 key 要多
    占 ~75 MB 临时内存，而这里只有 96 次 98k 元素的 bincount。
    """
    m = np.asarray(mask)
    if m.ndim != 3:
        raise ValueError(f'organ_zprofiles 要 [D,W,H] 的分割，收到 {m.shape}')
    if m.size and int(m.max()) > N_ORGANS:
        raise ValueError(f'mask 最大值 {int(m.max())} > {N_ORGANS} 类器官')
    prof = np.zeros((N_ORGANS, m.shape[0]), np.int32)
    for zi in range(m.shape[0]):
        counts = np.bincount(m[zi].ravel(), minlength=N_ORGANS + 1)
        prof[:, zi] = counts[1:N_ORGANS + 1]
    return prof


def _center_crop(image, mask, crop_size):
    """照抄上游 :96-122。"""
    x_min, y_min, z_min, x_max, y_max, z_max = _masks_to_boxes_3d(mask)[0].long()
    crop_d, crop_h, crop_w = (
        max(crop_size[0], z_max - z_min),
        max(crop_size[1], y_max - y_min),
        max(crop_size[2], x_max - x_min),
    )
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    cz = (z_min + z_max) // 2
    d, h, w = image.shape[-3:]
    x_start = max(0, cx - crop_w // 2)
    x_end = min(w, x_start + crop_w)
    if x_end - x_start < crop_w:
        x_start = max(0, x_end - crop_w)
    y_start = max(0, cy - crop_h // 2)
    y_end = min(h, y_start + crop_h)
    if y_end - y_start < crop_h:
        y_start = max(0, y_end - crop_h)
    z_start = max(0, cz - crop_d // 2)
    z_end = min(d, z_start + crop_d)
    if z_end - z_start < crop_d:
        z_start = max(0, z_end - crop_d)
    return (
        image[..., z_start:z_end, y_start:y_end, x_start:x_end],
        mask[..., z_start:z_end, y_start:y_end, x_start:x_end],
    )


@torch.inference_mode()
def forward_window(model: RadarVisual, images, organ_logits, test_organs, text_feat_dict,
                   organ_feat_dict, skip_organ=None):
    """照抄上游 `RADAR.forward_test_win`（:308-399）。organ_logits/organ_feat_dict 就地更新。"""
    seg_probs, seg, emb1, emb2, emb3, flags1, flags2, flags3 = model.visual_encoder(images, None)
    margin = 2
    masks = seg
    for i, (embed1, embed2, embed3, mask) in enumerate(zip(emb1, emb2, emb3, masks)):
        boundaries = []
        for d in range(mask.dim()):
            start_slice = [slice(None)] * mask.dim()
            end_slice = [slice(None)] * mask.dim()
            start_slice[d] = slice(None, margin)
            end_slice[d] = slice(-margin, None)
            boundaries.append(mask[tuple(start_slice)][mask[tuple(start_slice)] > 0])
            boundaries.append(mask[tuple(end_slice)][mask[tuple(end_slice)] > 0])
        boundaries = torch.cat(boundaries)
        boundary_values = boundaries[boundaries > 0].flatten()
        boundary_organs = torch.unique(boundary_values)
        if skip_organ is not None:
            boundary_organs = boundary_organs[boundary_organs != skip_organ + 1]

        organ_ids, organ_counts = torch.unique(mask, return_counts=True)
        organ_ids = organ_ids.long()
        organ_counts = organ_counts[organ_ids != 0]
        organ_ids = organ_ids[organ_ids != 0]

        intact_organ_ids = [o for o, c in zip(organ_ids, organ_counts) if o not in boundary_organs]
        intact_organ_ids = torch.tensor(intact_organ_ids, device=masks.device).long()
        intact_organ_ids = intact_organ_ids - 1
        if not len(intact_organ_ids):
            continue

        organ_sizes = dict(zip(
            [ORGANS[o] for o in intact_organ_ids],
            [organ_counts[organ_ids == o + 1].item() for o in intact_organ_ids],
        ))

        for organ_id in intact_organ_ids:
            organ_name = ORGANS[organ_id.item()]
            if organ_name not in test_organs:
                continue
            if organ_name in organ_feat_dict:
                continue
            tokens1 = flags1[i, organ_id, :]
            tokens2 = flags2[i, organ_id, :]
            tokens3 = flags3[i, organ_id, :]
            query = model.query_tokens[organ_id].unsqueeze(0).unsqueeze(0)
            key1 = embed1[tokens1].unsqueeze(0)
            key2 = embed2[tokens2].unsqueeze(0)
            key3 = embed3[tokens3].unsqueeze(0)
            key = value = torch.cat([key1, key2, key3], dim=1)
            updated_query_token, _ = model.attention(query, key, value)
            updated_query_token = updated_query_token.squeeze(0)
            image_feat = F.normalize(model.vision_projs[organ_id](updated_query_token), dim=-1)
            organ_feat_dict[organ_name] = image_feat.cpu()

            for item in organ_logits.keys():
                item_organ_name = item.split('_')[0] if isinstance(item, str) else item[0]
                if item_organ_name != organ_name:
                    continue
                text_feat = text_feat_dict[item]
                logits = image_feat @ text_feat.t() / model.temp
                probs = logits.softmax(-1)
                organ_logits[item].append(probs.cpu().tolist())
    return organ_logits, seg_probs


@torch.inference_mode()
def infer_volume(model: RadarVisual, image: torch.Tensor, test_items, text_feat_dict,
                 device: str = 'cpu', pad_func=None, compute_organs=None):
    """照抄上游 `evaluate` 的单例主体（:428-561，去掉 DataLoader/dist）。

    image: [1, C, D, W, H]（已预处理、已 padding）；单例时 batch=1。
    compute_organs: 要计算向量的器官名序列（cache 传全部 36 个）。默认 None =
        上游口径（只算 test_items 涉及的器官）。它只扩大 organ_feats 的**产出集合**，
        不改任何数值路径 —— 对拍（parity_demo）走默认分支，逐位结论不受影响。
    -> dict：
        organ_feats   : {中文器官名: tensor[256]}（L2-normalized，CPU）
        organ_sizes   : {中文器官名: int}
        stitched_mask : np.ndarray[D,W,H] uint8（37 类 argmax 拼接）
        probs         : {test_item: float}（窗口平均后的 pos 概率）
        sag_voxels    : float（胆囊体素数；上游用它当 surgically_absent 的输出列）
        n_windows     : int
    """
    assert image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1, (
        f'infer_volume 要 [1,1,D,W,H]，拿到 {tuple(image.shape)}'
        '（load_and_prepare 返回 4D [C,D,W,H]，调用方加 batch 维）'
    )
    image = image.to(device)
    if compute_organs is None:
        test_organs = list({item.split('_')[0] for item in test_items})
    else:
        test_organs = list(compute_organs)

    image_size = list(image.shape[2:])
    num_spatial_dims = len(image.shape) - 2
    scan_interval = _get_scan_interval(image_size, ROI_SIZE, num_spatial_dims, OVERLAP)
    slices = dense_patch_slices(image_size, ROI_SIZE, scan_interval)
    num_win = len(slices)

    organ_logits = dict(zip(test_items, [[] for _ in test_items]))
    organ_logits.pop('胆囊_术后胆囊缺失', None)
    organ_feat_dict: dict[str, torch.Tensor] = {}

    full_mask = torch.zeros((1, 37) + tuple(image_size), device=device)
    count_map = torch.zeros_like(full_mask)

    for slice_g in range(0, num_win, SW_BATCH):
        slice_range = range(slice_g, min(slice_g + SW_BATCH, num_win))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_patches = torch.cat([image[win_slice] for win_slice in unravel_slice]).to(device)
        organ_logits, pred_window_seg_prob = forward_window(
            model, window_patches, organ_logits, test_organs, text_feat_dict, organ_feat_dict)
        interpolated_seg_prob = F.interpolate(pred_window_seg_prob, size=window_patches.shape[2:], mode='trilinear')
        for ii, slice_idx in enumerate(slice_range):
            full_slice = unravel_slice[ii]
            full_mask[full_slice] += interpolated_seg_prob[ii]
            count_map[full_slice] += 1

    count_map = torch.clamp(count_map, min=1)
    stitched_mask = (full_mask / count_map).argmax(1).unsqueeze(0)

    margin = 2
    boundaries = []
    squeeze = stitched_mask.squeeze(0).squeeze(0)
    for d in range(squeeze.dim()):
        start_slice = [slice(None)] * squeeze.dim()
        end_slice = [slice(None)] * squeeze.dim()
        start_slice[d] = slice(None, margin)
        end_slice[d] = slice(-margin, None)
        boundaries.append(squeeze[tuple(start_slice)][squeeze[tuple(start_slice)] > 0])
        boundaries.append(squeeze[tuple(end_slice)][squeeze[tuple(end_slice)] > 0])
    boundaries = torch.cat(boundaries)
    boundary_organs = torch.unique(boundaries[boundaries > 0])

    for k, v in list(organ_logits.items()):
        if len(v):
            continue
        organ_name = k.split('_')[0]
        organ_id = ORGANS.index(organ_name)
        window_patch, window_mask = _center_crop(
            image, torch.eq(stitched_mask, organ_id + 1), crop_size=ROI_SIZE)
        window_mask = window_mask.float()
        window_mask[window_mask == 1] = organ_id + 1
        pad_data = pad_func({'image': window_patch[0], 'label': window_mask[0]})
        window_patch, window_mask = pad_data['image'], pad_data['label']
        organ_logits, _ = forward_window(
            model, window_patch[None].to(device), organ_logits, test_organs, text_feat_dict,
            organ_feat_dict, skip_organ=organ_id)

    if GALLBLADDER_ID not in torch.unique(stitched_mask):
        sag_prob = 0.
    else:
        sag_prob = (stitched_mask == GALLBLADDER_ID).sum().item()

    probs = {}
    for item, plist in organ_logits.items():
        if not len(plist):
            continue
        probs[item] = float(np.concatenate(plist).mean(0)[1])

    organ_sizes = {}
    mask_cpu = stitched_mask.squeeze(0).squeeze(0).cpu()
    ids, counts = torch.unique(mask_cpu, return_counts=True)
    for oid, cnt in zip(ids.tolist(), counts.tolist()):
        if oid == 0:
            continue
        organ_sizes[ORGANS[oid - 1]] = int(cnt)

    return {
        'organ_feats': {k: v.float() for k, v in organ_feat_dict.items()},
        'organ_sizes': organ_sizes,
        'stitched_mask': mask_cpu.numpy().astype(np.uint8),
        'probs': probs,
        'sag_voxels': float(sag_prob),
        'n_windows': int(num_win),
    }


@torch.inference_mode()
def forward_no_window(model: RadarVisual, image, compute_organs, text_feat_dict=None,
                      organ_logits=None):
    """A2 的器官支路 —— `forward_window` 的**去滑窗**版本：单次前向，逐器官池化。

    与 `forward_window` 逐处对照（差别的来源只有一个：这里 `image` 就是整卷，
    不是一个 ROI 窗口）：

    * 编码器调用、三尺度 token、`query_tokens[organ_id]` + `attention` +
      `vision_projs[organ_id]` + L2 归一化 —— **全部原样**，一个算子都没换。
      权重也是同一份（见 `ENCODER_IDENTITIES` 的 `radar_ft_nowin`）。
    * 触边规则**保留**：与窗口那一路一样，mask 里贴到 2 体元边框的器官被跳过
      （`intact_organ_ids`）。但「边」现在是**整卷**的边，而窗口那一路的边是每个 ROI
      的内部边界 —— 同一条规则在两种几何下选出的器官集合不同，这是 A2 要付的账，
      如实登记，不假装它等价。
    * `_center_crop` 兜底**没有**：滑窗那条路里「没有任何窗口命中该器官」是一个
      可修的事件（换个视野再评一次），这里单次前向就是全部视野 —— 没命中就是不在了。
    * `flags*` 由网络自己的 argmax mask 池化到三个 token 网格（`vision_branch.py:126-158`），
      与窗口那一路同一份实现；变的只是被池化的那张 mask 覆盖的范围。

    本函数**故意不复用** `forward_window` 的代码：后者受 `parity_demo.py` 的逐位对拍
    约束（上游 CSV 是它的锚），任何重构都把它从「被验过的东西」变成「要重验的东西」。
    两处相加的那 40 行是这个仓库里比较便宜的一种重复。

    image `[1,1,D,W,H]`（已按 `volume.load_and_prepare_whole` 定形）
    -> `(organ_feat_dict, seg_probs)`：`{中文器官名: tensor[1,256] CPU}`，与
       `forward_window` 写进 `organ_feat_dict` 的东西同形状同语义。
    """
    if image.ndim != 5 or image.shape[0] != 1 or image.shape[1] != 1:
        raise ValueError(f'forward_no_window 要 [1,1,D,W,H]，拿到 {tuple(image.shape)}')
    seg_probs, seg, emb1, emb2, emb3, flags1, flags2, flags3 = model.visual_encoder(image, None)
    margin = 2
    mask = seg[0]
    boundaries = []
    for d in range(mask.dim()):
        start_slice = [slice(None)] * mask.dim()
        end_slice = [slice(None)] * mask.dim()
        start_slice[d] = slice(None, margin)
        end_slice[d] = slice(-margin, None)
        boundaries.append(mask[tuple(start_slice)][mask[tuple(start_slice)] > 0])
        boundaries.append(mask[tuple(end_slice)][mask[tuple(end_slice)] > 0])
    boundaries = torch.cat(boundaries)
    boundary_organs = torch.unique(boundaries[boundaries > 0])

    organ_ids, organ_counts = torch.unique(mask, return_counts=True)
    organ_ids = organ_ids.long()
    organ_counts = organ_counts[organ_ids != 0]
    organ_ids = organ_ids[organ_ids != 0]
    intact_organ_ids = [o for o, c in zip(organ_ids, organ_counts) if o not in boundary_organs]
    intact_organ_ids = torch.tensor(intact_organ_ids, device=mask.device).long()
    intact_organ_ids = intact_organ_ids - 1

    want = set(compute_organs)
    feats: dict[str, torch.Tensor] = {}
    for organ_id in intact_organ_ids:
        organ_name = ORGANS[organ_id.item()]
        if organ_name not in want:
            continue
        tokens1 = flags1[0, organ_id, :]
        tokens2 = flags2[0, organ_id, :]
        tokens3 = flags3[0, organ_id, :]
        key1 = emb1[tokens1].unsqueeze(0)
        key2 = emb2[tokens2].unsqueeze(0)
        key3 = emb3[tokens3].unsqueeze(0)
        key = value = torch.cat([key1, key2, key3], dim=1)
        updated_query_token, _ = model.attention(
            model.query_tokens[organ_id].unsqueeze(0).unsqueeze(0), key, value)
        updated_query_token = updated_query_token.squeeze(0)
        image_feat = F.normalize(model.vision_projs[organ_id](updated_query_token), dim=-1)
        feats[organ_name] = image_feat.cpu()
        if organ_logits is None:
            continue
        for item in organ_logits.keys():
            item_organ_name = item.split('_')[0] if isinstance(item, str) else item[0]
            if item_organ_name != organ_name:
                continue
            probs = (image_feat @ text_feat_dict[item].t() / model.temp).softmax(-1)
            organ_logits[item].append(probs.cpu().tolist())
    return feats, seg_probs


@torch.inference_mode()
def infer_volume_nowin(model: RadarVisual, image: torch.Tensor, test_items, text_feat_dict,
                       device: str = 'cpu', compute_organs=None):
    """A2 的单例主体 —— `infer_volume` 去掉滑窗后的对应物，返回**同构的 res**。

    返回值与 `infer_volume` 逐键同名（`save_npz` 因此一份实现两用），语义上有三处必须
    登记的差别：

    1. `n_windows` 恒为 1（这就是本臂的定义）。
    2. `stitched_mask` 在**整图支路那一卷**的坐标系里（`load_and_prepare_whole` 定形的
       96×256×384），而滑窗那条路的 stitched_mask 在 `load_and_prepare`（非零区 crop +
       DivisiblePad，逐卷变尺寸）的坐标系里。⇒ 由 mask 现算的那四列
       （`organ_box` / `organ_zprof` / `sizes` / `sag_voxels`）**不能**跟过来用：切片缓存
       与证据 token 的 K/V 读的是后一个坐标系（`slice_frontend.py:10`），拿这一卷的 box
       去框那一卷的层 = 每个器官的证据都来自别人的位置，前向不报错、读数全假。
       所以 A2 的缓存在编码时把这四列从滑窗缓存**原样搬过来**（`cache.py --geom-from-root`），
       meta 记 `geom_frame='window'`。本臂由此退化成真正的单变量：**只有 36 个器官向量
       换了函数**，几何、切片、题面、步数一律与旗舰同源。
    3. 上游的「padded 任一维 >1000 就不做滑窗」跳过规则在这里**不适用**（单次前向
       没有窗口可跳，输入又被 `load_and_prepare_whole` 定形）。⇒ 那批 study 在旗舰上是
       `has_feat` 全 False（走可学 missing 槽），在本臂上有真向量。这是覆盖率差异，
       不是等价，run_info / 片尾汇总要把例数记出来。

    stitched 的算法与滑窗那条路在「恰好一个窗口、覆盖全卷」时逐位相同：那边是
    `full_mask += interpolate(seg_probs)`、`count_map += 1`，再 `(full/count).argmax(1)`；
    一次累加除以 1 就是原值，所以这里直接 `interpolate(seg_probs).argmax(1)`。
    """
    assert image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1, (
        f'infer_volume_nowin 要 [1,1,D,W,H]，拿到 {tuple(image.shape)}'
        '（volume.load_and_prepare_whole 返回 4D [C,D,W,H]，调用方加 batch 维）')
    image = image.to(device)
    if compute_organs is None:
        organs = list({item.split('_')[0] for item in test_items})
    else:
        organs = list(compute_organs)

    organ_logits = dict(zip(test_items, [[] for _ in test_items]))
    organ_logits.pop(SAG_ITEM, None)
    feats, seg_probs = forward_no_window(
        model, image, organs, text_feat_dict,
        organ_logits if len(organ_logits) else None)

    stitched_mask = F.interpolate(
        seg_probs, size=tuple(image.shape[2:]), mode='trilinear').argmax(1).unsqueeze(0)

    if GALLBLADDER_ID not in torch.unique(stitched_mask):
        sag_prob = 0.
    else:
        sag_prob = (stitched_mask == GALLBLADDER_ID).sum().item()

    probs = {}
    for item, plist in organ_logits.items():
        if not len(plist):
            continue
        probs[item] = float(np.concatenate(plist).mean(0)[1])

    organ_sizes = {}
    mask_cpu = stitched_mask.squeeze(0).squeeze(0).cpu()
    ids, counts = torch.unique(mask_cpu, return_counts=True)
    for oid, cnt in zip(ids.tolist(), counts.tolist()):
        if oid == 0:
            continue
        organ_sizes[ORGANS[oid - 1]] = int(cnt)

    return {
        'organ_feats': {k: v.float() for k, v in feats.items()},
        'organ_sizes': organ_sizes,
        'stitched_mask': mask_cpu.numpy().astype(np.uint8),
        'probs': probs,
        'sag_voxels': float(sag_prob),
        'n_windows': 1,
    }


def default_pad_func():
    from monai import transforms
    return transforms.DivisiblePadd(
        keys=["image", "label"], k=32, mode='constant', constant_values=0, method="end")


@torch.inference_mode()
def infer_whole(model: 'RadarPlusVisual', image: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """整图分支 —— 照抄上游 `RADAR_train/infer_whole.py:671-678`，逐行对得上。

    与 `forward_window` 的三处实质差别：
      * key/value = **全部** patch token（`cat([emb1,emb2,emb3], dim=1)`），
        不做触边筛选、不按器官取 flags —— 器官那一路是 `key = embed[tokens]`；
      * query = `query_tokens_whole`（1 个 token，expand 到 batch）；
      * 单次前向、**没有滑窗**：输入已由 `load_and_prepare_whole` 定形到
        (96,256,384)，所以每卷恰好评一次、没有窗口平均，也不存在"某器官没命中"。
    -> tensor[256]（L2-normalized，CPU）
    """
    assert image.ndim == 5 and image.shape[0] == 1 and image.shape[1] == 1, (
        f'infer_whole 要 [1,1,D,W,H]，拿到 {tuple(image.shape)}')
    if tuple(image.shape[2:]) != ROI_SIZE:
        raise ValueError(
            f'整图支路输入必须是定形 {ROI_SIZE}（pad+center crop 的产物），拿到 '
            f'{tuple(image.shape[2:])} —— 用 volume.load_and_prepare_whole，'
            '不要拿器官支路那卷非零区 crop 的结果凑')
    if not hasattr(model, 'attention_whole'):
        raise ValueError('整图分支需要 RadarPlusVisual（radar_ft 身份）；'
                         'pretrain 权重里没有 *_whole（C.43 ② 实测 0 个张量）')
    image = image.to(device)
    seg_probs, seg, emb1, emb2, emb3, _f1, _f2, _f3 = model.visual_encoder(image, None)
    key_whole = value_whole = torch.cat([emb1, emb2, emb3], dim=1)
    query_whole = model.query_tokens_whole.unsqueeze(0).expand(key_whole.shape[0], -1, -1)
    feat, _ = model.attention_whole(query_whole, key_whole, value_whole)
    feat = feat.squeeze(1)
    feat = model.vision_projs_whole(feat)
    feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0).cpu()

