"""RADAR 体数据预处理 —— 逐字复用上游 `inference_source_testset.py` 的 DataFolder 链路。

出处（行号以 /tmp/damo-radar@main 为准）：
    LoadImaged/affine 提取        :174-184
    spacing -> target_size 交叉写法 :185-190
    Resized(trilinear)+Transposed  :192-200
    clip HU [-300,400] + 全体积 min-max :202-204
    非零区 crop（外扩 d=5,hw=20） :207-230
    SpatialPadd 到 (96,256,384)    :143-149, 233-235

整图支路（`load_and_prepare_whole`）另出一套 —— 上游是**另一个脚本**
`RADAR_train/infer_whole.py`，除了「Load + 交叉索引 resize」这段完全相同，
后面全都不同（:428-453, 505-531）：先 pad 再 CenterSpatialCrop 定形 (96,256,384)，
**然后**才 clip 到 [-1000,1000]，且**没有**非零区 crop、**没有**滑窗。
两支不是同一个函数的两种写法，是两个不同的输入分布 —— 别互相"顺手统一"。

三条不许"顺手修正"的点（改了就不是上游那套预处理，对拍会当场对不上）：
1. `target_size = [int(h*scale[1]), int(w*scale[0]), int(d*scale[2])]` —— 索引交叉，
   见 geometry._dims_after_resample 的注释。
2. 全体积（非逐切片）min-max；`image.min()` 在 crop 之前、clip 之后。
3. `image[image > 400] = 400` 是**先 clip 再归一化**，且 ROI 之外的 0 值体素
   也参与 min/max（上游如此）。
4. 整图支路的归一化是 `(x-min)/(max-min)`，**没有** 器官支路那个 `+1e-8`
   （whole:530 vs testset:204）—— 逐字照抄，不补。

本模块只做预处理，不碰模型；返回 [C,D,W,H] 的 float 张量与 crop 元数据。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from monai import transforms

REF_SPACING = (1.0, 1.0, 5.0)
PAD_SIZE = (96, 256, 384)
HU_LO, HU_HI = -300.0, 400.0
EXTEND_D, EXTEND_HW = 5, 20
HU_LO_WHOLE, HU_HI_WHOLE = -1000.0, 1000.0


def _pad_func():
    return transforms.SpatialPadd(
        keys=["image"],
        spatial_size=PAD_SIZE,
        mode="constant",
        constant_values=0,
        method="end",
    )


def _center_crop_func():
    """整图支路独有：定形到 PAD_SIZE（上游 :450-453 与 pad_func 同尺寸）。"""
    return transforms.CenterSpatialCropd(keys=["image"], roi_size=PAD_SIZE)


def _load_resized(image_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """两支共同的前半段：Load -> 交叉索引 resize -> Transposed -> [C, D', W', H']。"""
    data = {"image": image_path}
    res = transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True)(data)
    image = res["image"]
    affine = res["image_meta_dict"]["affine"]
    spacing = tuple(abs(affine[i, i].item()) for i in range(3))
    _, h, w, d = image.shape
    orig_shape_hwd = (int(h), int(w), int(d))

    scale = [spacing[i] / REF_SPACING[i] for i in range(3)]
    target_size = [int(h * scale[1]), int(w * scale[0]), int(d * scale[2])]

    trans = transforms.Compose(
        [
            transforms.Resized(spatial_size=target_size, keys=["image"], mode="trilinear"),
            transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        ]
    )
    image = trans(res)["image"]

    meta = {
        "orig_shape_hwd": orig_shape_hwd,
        "resample_target_hwd": [int(t) for t in target_size],
        "spacing": [float(s) for s in spacing],
    }
    return image, meta


def load_and_prepare(image_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """器官支路 -> (image[C,D,W,H] padded to (96,256,384), meta)

    meta: orig_shape_hwd / resample_target_hwd / crop_shape_dhw / padded_shape_dhw / spacing
    """
    image, meta = _load_resized(image_path)

    image[image > HU_HI] = HU_HI
    image[image < HU_LO] = HU_LO
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    roi_coords = np.nonzero(image[0].cpu().numpy())
    min_dhw = torch.from_numpy(np.min(roi_coords, axis=1))
    max_dhw = torch.from_numpy(np.max(roi_coords, axis=1))

    min_dhw = torch.max(
        min_dhw - torch.tensor([EXTEND_D, EXTEND_HW, EXTEND_HW]),
        torch.tensor([0, 0, 0]),
    )
    max_dhw = torch.min(
        max_dhw + torch.tensor([EXTEND_D, EXTEND_HW, EXTEND_HW]),
        torch.tensor([image.shape[1], image.shape[2], image.shape[3]]),
    )

    cropped = image[:, min_dhw[0]: max_dhw[0], min_dhw[1]: max_dhw[1], min_dhw[2]: max_dhw[2]]
    meta["crop_shape_dhw"] = tuple(int(s) for s in cropped.shape[1:])

    out = _pad_func()({"image": cropped})["image"].as_tensor()
    meta["padded_shape_dhw"] = tuple(int(s) for s in out.shape[1:])
    return out, meta


def load_and_prepare_whole(image_path: str) -> tuple[torch.Tensor, dict[str, Any]]:
    """整图支路（infer_whole.py:493-544）-> (image[1,96,256,384], meta)

    顺序是上游的顺序：resize -> **pad -> center crop** -> clip -> 归一化。
    与非零区 crop 那一路的差别不只是窗宽：先定形再窗，等于把「ROI 之外补的 0」
    也算进 min/max（上游如此），并且长卷会被中心裁掉两端。
    """
    image, meta = _load_resized(image_path)

    data = _pad_func()({"image": image})
    data = _center_crop_func()(data)
    image = data["image"]

    image[image > HU_HI_WHOLE] = HU_HI_WHOLE
    image[image < HU_LO_WHOLE] = HU_LO_WHOLE
    image = (image - image.min()) / (image.max() - image.min())

    out = image.as_tensor() if hasattr(image, "as_tensor") else image
    shape_dhw = tuple(int(s) for s in out.shape[1:])
    if shape_dhw != PAD_SIZE:
        raise RuntimeError(f'整图支路 pad+center crop 后应为 {PAD_SIZE}，拿到 {shape_dhw}')
    meta["whole_shape_dhw"] = shape_dhw
    return out, meta
