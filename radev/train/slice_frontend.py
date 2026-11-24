"""B 支的第二条视觉通路：体数据 -> N 张切片 -> 宿主自己的 processor 张量。

计划出处：`.qoder/plans/earnest-mesa-source.md` 的 B 支（原生多模态宿主 + RADAR 附加编码器）。
与 `visual_frontend.py` 是**并列**的两条通路，不是它的替代：那边出 36/37 个器官向量走
projector，这边出 `pixel_values`/`image_grid_thw` 走宿主自带的视觉塔。序列里两种插槽
各占各的 token id，接线见 `model.py` 的 `_embed`。

三条口径要说明白（都不是防御性编程，是这套数据特有的坑）：

1. **切片来自 RADAR 那一路预处理后的体积**（`volume.load_and_prepare`：非零区 crop +
   pad 到 (96,256,384) + HU [-300,400] + 全体积 min-max 归一）。理由不是省事 —— 两条
   通道看的是同一个 ROI、同一个窗，「视觉有贡献」的读数才不会被「一个看了 ROI 一个看了
   全卷」这种预处理差异污染。代价是原生塔看到的是被 anisotropic resize 过的图，
   与它预训练时的自然图像分布差得更远；这条**必须在消融里单独立项**，不能默认无事。
2. **off 臂的切片下标是 `linspace` 的确定性均匀采样**，不依数据（不做「挑有病灶的层」）；
   `--organ-crop mask` 是显式对立的器官先验臂：按题绑定器官的 stitched-mask 逐层体素
   剖面选层。挑层等于把一维定位信息变成先验，所以两臂必须各自立项、同参数对照，
   不能把 mask 臂的读数当成模型自己学会了定位。
3. **盘上缓存存 uint8 切片，不存 `pixel_values`**。processor 的输出形状由 config 的
   patch/merge 定，换宿主就作废；uint8 切片与宿主无关，且比 fp32 张量小 **25 倍**
   （32 片 345×410：4.5 MB vs 112.5 MB）。processor 每 epoch 重跑一遍，代价是 CPU 时间，
   换的是「换模型不用重刷缓存」。
   **RAM 里另有一层 processed 备忘**（`proc_lru>0`，见 `_proc_cached`）—— 它不落盘，
   所以「换宿主不用重刷」这条不受影响；它只是不再为同一卷的第 2..n 题重跑 processor。
   缺省 `proc_lru=0` = 逐位是老行为。

`num_workers` 必须是 0 —— 不是这里的要求（本模块纯 CPU），而是同一个 batch 里
`VisualFrontend` 的要求（它的塔在 CUDA 上）。两个前端共用 dataset 的取数入口，
所以口径跟着紧的那一个。
"""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from .. import config
from . import volume

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(REPO, 'out_slice_cache')


def pick_indices(n_avail: int, n_want: int) -> np.ndarray:
    """确定性均匀取层：`linspace(0, n_avail-1, n_want)` 四舍五入去重后必须正好 n_want 个。

    去重后变少 = 想取的片数比可用的层还多 —— 这不是「少取几片」的小问题，它会让
    `n_slices` 与实际喂进去的片数不一致，而序列里的占位数是按 `n_slices` 拼的。
    """
    if n_want < 1:
        raise ValueError(f'切片数要 >=1，实为 {n_want}')
    idx = np.unique(np.round(np.linspace(0, n_avail - 1, n_want)).astype(int))
    if len(idx) != n_want:
        raise ValueError(f'体积只有 {n_avail} 层，取不下 {n_want} 片（去重后剩 {len(idx)}）'
                         f'—— 降 --slices 或别裁这么狠')
    return idx


def crop_select(idx: np.ndarray, z0: int, z1: int, n_slices: int,
                stats: dict | None = None) -> np.ndarray:
    """缓存层位 `idx` 里落在 [z0,z1] 的部分 → 取 n_slices 片的**位置**下标（非层号）。

    确定性（无随机）：round(linspace(0, len-1, n_slices))，与 `pick_indices` 同一族取法，
    但这里**允许重复**——器官的 z 跨度可能比缓存网格还窄（默认 24/96 层 ≈ 4 层/格），
    重复层也好过缺片：片数恒 n_slices，序列占位数才不裂。空集（器官 z 范围与缓存网格
    错开）→ 距盒中心最近 1 层兜底。两个兜底各计一笔（crop_widen / crop_repeat），
    逐题可见，不静默。
    """
    if n_slices < 1:
        raise ValueError(f'n_slices={n_slices} < 1')
    idx = np.asarray(idx)
    pos = np.nonzero((idx >= int(z0)) & (idx <= int(z1)))[0]
    if pos.size == 0:
        pos = np.array([int(np.argmin(np.abs(idx.astype(np.float64) - (z0 + z1) / 2.0)))])
        if stats is not None:
            stats['crop_widen'] = stats.get('crop_widen', 0) + 1
    sel = np.round(np.linspace(0, pos.size - 1, n_slices)).astype(int)
    if np.unique(sel).size < n_slices and stats is not None:
        stats['crop_repeat'] = stats.get('crop_repeat', 0) + 1
    return pos[sel]


def mask_select(idx: np.ndarray, occ: np.ndarray, z0: int, z1: int, n_slices: int,
                stats: dict | None = None) -> np.ndarray:
    """按题绑定器官的逐层体素数选 n_slices 片 -> 缓存位置下标（非层号）。

    `occ[D]` 是 `organ_zprof` 对绑定器官的并集剖面（stitched mask 每层的体素数），
    `idx` 是同一 padded 体积缓存的层号。规则（确定性）：
      1. 只看 [z0,z1]（器官 bbox ± margin）内、且 `occ[idx] > 0` 的缓存层；
      2. 正层 >= n_slices：在正层序列上均匀取 n_slices —— 覆盖器官全长，不挑峰值层；
      3. 正层 < n_slices：全取正层，余量按「离最近正层的 z 距离」补；距离相同取离
         正层中心近的；
      4. 一个正层都没有：回退 `crop_select`（bbox 内均匀取层），计 `mask_empty`。
    输出恒 n_slices 个位置；缓存层数不足 n_slices 时回退 `crop_select`（允许重复）。
    """
    if n_slices < 1:
        raise ValueError(f'n_slices={n_slices} < 1')
    idx = np.asarray(idx)
    occ = np.asarray(occ).reshape(-1)
    if idx.size == 0:
        raise ValueError('mask_select 收到空 idx')
    if occ.size <= int(idx.max()):
        raise ValueError(f'organ_zprof 只有 {occ.size} 层，缓存 idx 最大 {int(idx.max())}'
                         ' —— 剖面与切片不是同一份 padded 体积')
    if n_slices > idx.size:
        if stats is not None:
            stats['mask_short'] = stats.get('mask_short', 0) + 1
        return crop_select(idx, z0, z1, n_slices, stats)
    in_box = (idx >= int(z0)) & (idx <= int(z1))
    pos = np.nonzero(in_box & (occ[idx] > 0))[0]
    if pos.size == 0:
        if stats is not None:
            stats['mask_empty'] = stats.get('mask_empty', 0) + 1
        return crop_select(idx, z0, z1, n_slices, stats)
    if stats is not None:
        stats['mask_pick'] = stats.get('mask_pick', 0) + 1
    if pos.size >= n_slices:
        pick = np.round(np.linspace(0, pos.size - 1, n_slices)).astype(int)
        return pos[pick]
    fill_n = n_slices - pos.size
    rest = np.setdiff1d(np.arange(idx.size), pos, assume_unique=True)
    dist = np.abs(idx[rest][:, None] - idx[pos][None, :]).min(axis=1)
    center = float(idx[pos].mean())
    order = np.lexsort((np.abs(idx[rest] - center), dist))
    sel = np.concatenate([pos, rest[order[:fill_n]]])
    sel.sort()
    if stats is not None:
        stats['mask_fill'] = stats.get('mask_fill', 0) + 1
    return sel


def union_crop(rows: np.ndarray, margin: float = 0.1, w: int = 256, h: int = 384,
               min_edge: int = 32):
    """organ_box 行集 → 并集 bbox `(z0,z1,y0,y1,x0,x1)`（含界）；全无效 → None。

    行 [k,6] int16，坐标序与 `radar_encoder.organ_boxes` 一致（z=dim0、y=dim1=w、x=dim2=h）；
    含 -1 的行被剔除（mask 里没该类）。margin 外扩口径：每边 round(margin×该轴跨度)，
    z 每边至少 1 层、**不做 clamp**（z 的合法性由取层侧天然保证：只从缓存层位里选）；
    平面轴不足最小边 min_edge 时对称补足（贴边向内移），再 clamp 回 [0, w-1]/[0, h-1]。
    纯函数、两次调用逐位同 —— smoke [8] 的确定性断言盯的就是这个。
    """
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 6:
        raise ValueError(f'union_crop 要 [k,6] 的行集，收到 {rows.shape}')
    ok = rows[(rows >= 0).all(axis=1)]
    if ok.size == 0:
        return None
    z0, z1 = _expand_edge(int(ok[:, 0].min()), int(ok[:, 1].max()), margin,
                          ext_min=1, min_edge=1, cap=None)
    y0, y1 = _expand_edge(int(ok[:, 2].min()), int(ok[:, 3].max()), margin,
                          ext_min=0, min_edge=min_edge, cap=w - 1)
    x0, x1 = _expand_edge(int(ok[:, 4].min()), int(ok[:, 5].max()), margin,
                          ext_min=0, min_edge=min_edge, cap=h - 1)
    return z0, z1, y0, y1, x0, x1


def _expand_edge(lo: int, hi: int, margin: float, *, ext_min: int, min_edge: int,
                 cap: int | None):
    """单轴：按比例外扩 → 补最小边 → cap 内平移/截断（cap=None 则只外扩）。"""
    ext = max(int(ext_min), int(round(float(margin) * (hi - lo))))
    lo, hi = lo - ext, hi + ext
    if min_edge > 1 and hi - lo + 1 < min_edge:
        need = min_edge - (hi - lo + 1)
        lo -= need // 2
        hi += need - need // 2
    if cap is None:
        return lo, hi
    if lo < 0:
        hi = min(cap, hi - lo)
        lo = 0
    if hi > cap:
        lo = max(0, lo - (hi - cap))
        hi = cap
    return lo, hi


class SliceFrontend:
    """`sid -> (pixel_values [ΣP, patch_dim] float32, image_grid_thw [N,3] int64)`。

    `host_path` 是宿主目录（Qwen3.5-0.8B/2B/4B），processor 从它读 —— **processor 与权重
    必须同目录**：patch/merge/mean-std 都写在那份 config 里，混用会得到 grid 与占位数
    不一致的序列。

    `organ_crop != 'off'`（B 支器官裁剪臂）：多一条**按题取层分支** —— 缓存整卷全幅
    `<Mreq>c_`（Mreq=max(n_slices, crop_cache_n)），取数时按题绑定器官的定位列选层：
    `z`=bbox 的 z 范围（平面全幅）、`bbox`=z 范围+平面裁、`mask`=逐层器官体素剖面
    （平面全幅）。读数都是「器官先验（+高分辨率）」的合成增益（与上游 `_center_crop`
    同一性质），**不是**模型自己学会定位。该臂的对照 = 同参数 `--organ-crop off`。
    `crop=None`（本卷缺 box / 组无类目）逐题回退 off 文件与 off 口径，计数 crop_nobox。
    """

    def __init__(self, host_path: str, n_slices: int, *, root: str = DEFAULT_ROOT,
                 write: bool = True, lru: int = 8, proc_lru: int = 0,
                 organ_crop: str = 'off',
                 crop_margin: float = 0.1, crop_cache_n: int = 24) -> None:
        from transformers import AutoConfig, AutoProcessor
        self.n_slices = int(n_slices)
        if self.n_slices < 1:
            raise ValueError(f'n_slices={n_slices} < 1 —— 关掉原生通道请走 --slices 0，'
                             '不要构造一个空前端')
        self.organ_crop = str(organ_crop)
        if self.organ_crop not in ('off', 'z', 'bbox', 'mask'):
            raise ValueError(f'organ_crop={organ_crop!r} 不在 off/z/bbox/mask')
        self.crop_margin = float(crop_margin)
        self.crop_cache_n = int(crop_cache_n)
        self.mreq = (self.n_slices if self.organ_crop == 'off'
                     else max(self.n_slices, self.crop_cache_n))
        self.host_path = host_path
        self.root = root
        self.write = bool(write)
        self.lru = int(lru)
        self.proc_lru = int(proc_lru)
        self.proc = AutoProcessor.from_pretrained(host_path)
        cfg = AutoConfig.from_pretrained(host_path)
        self.img_ids = (int(cfg.vision_start_token_id), int(cfg.image_token_id),
                        int(cfg.vision_end_token_id))
        self.merge = int(cfg.vision_config.spatial_merge_size)
        self.stats = {'hit_mem': 0, 'hit_disk': 0, 'load': 0,
                      'hit_proc': 0, 'miss_proc': 0,
                      'crop_ok': 0, 'crop_nobox': 0, 'crop_widen': 0, 'crop_repeat': 0,
                      'mask_pick': 0, 'mask_fill': 0, 'mask_empty': 0,
                      'mask_short': 0, 'mask_noprof': 0}
        self._mem: OrderedDict[tuple, tuple] = OrderedDict()
        self._pmem: OrderedDict[tuple, tuple] = OrderedDict()
        self.info = {'processor': type(self.proc).__name__, 'host_model_type': cfg.model_type,
                     'patch_size': int(cfg.vision_config.patch_size),
                     'merge_size': self.merge, 'img_ids': self.img_ids}

    def path(self, split: str, sid: str) -> str:
        """本臂的切片缓存路径：off → `<n>s_`（历史命名，逐位不变）；裁剪 → `<Mreq>c_`。"""
        if self.organ_crop == 'off':
            return self._path_off(split, sid)
        return os.path.join(self.root, split, f'{self.mreq}c_{sid}.npz')

    def _path_off(self, split: str, sid: str) -> str:
        return os.path.join(self.root, split, f'{self.n_slices}s_{sid}.npz')

    def pixels(self, sid: str, split: str,
               crop: tuple | None = None, *,
               with_layers: bool = False, vol: str | None = None):
        """三级：RAM LRU -> 盘上 uint8 切片 -> 现读体积。返回 processor 的产物。

        `crop=(z0,z1,y0,y1,x0,x1[,occ])`（organ_crop 非 off 时由 dataset 按题绑定器官算）：
        从 `<Mreq>c_` 全幅缓存里按模式选层；`bbox` 额外平面裁剪，`mask` 用第 7 位
        逐层器官体素剖面选层（见 `crop_select`/`mask_select`/`union_crop`）。
        `crop=None`：off 口径（`<n>s_` 全幅 n_slices 片）—— organ_crop 非 off 时这是
        逐题回退（无映射/组无类目/卷缺 box），计数 crop_nobox，输出与 off 臂逐位相同。

        `with_layers=True`（`--organ2d` 用）：多返回**每片在 padded 体积里的层号**
        （off 口径 = 缓存 `indices`；裁剪口径 = `indices[选中位置]`）。空间归属的 z
        判定要的是层号不是位置序 —— off 口径下两数恰好相同，裁剪口径下不同。

        `vol`：数据集记录自带的体积图像路径。缺省按 sid 拼 `image_path(sid)`
        （IID 口径）；拼出的文件不存在（外源 sid）时才用 `vol` 兜底 ——
        IID 下两条路径指向同一文件，行为逐位不变。
        """
        if crop is None:
            if self.organ_crop != 'off':
                self.stats['crop_nobox'] += 1
            pv, grid, layers = self._pixels_off(sid, split, vol)
        else:
            pv, grid, layers = self._pixels_crop(sid, split, crop, vol)
        if with_layers:
            return pv, grid, layers
        return pv, grid

    def _pixels_off(self, sid: str, split: str, vol: str | None = None) -> tuple:
        key = (sid, 's')
        m = self._mem.get(key)
        if m is not None:
            self._mem.move_to_end(key)
            self.stats['hit_mem'] += 1
            return (*self._proc_cached(key, m[0]), m[1])
        p = self._path_off(split, sid)
        if not os.path.isfile(p):
            slices, idx = self._from_volume(sid, split, self.n_slices, vol=vol)
            if self.write:
                self._save(p, slices, idx)
            self._put(key, slices, idx)
            self.stats['load'] += 1
            return (*self._proc_cached(key, slices), idx)
        with np.load(p) as z:
            sl, idx = z['slices'], z['indices']
            if sl.shape[0] != self.n_slices or len(idx) != self.n_slices:
                raise RuntimeError(f'{p} 里是 {sl.shape[0]} 片（indices {len(idx)} 个），'
                                   f'本次要 {self.n_slices} 片 —— 缓存名带切片数'
                                   f'（`<n>s_<sid>.npz`）但仍撞上，说明目录串了')
            self._put(key, sl, idx)
            self.stats['hit_disk'] += 1
        return (*self._proc_cached(key, sl), idx)

    def _pixels_crop(self, sid: str, split: str, crop: tuple,
                     vol: str | None = None) -> tuple:
        if self.organ_crop == 'off':
            raise RuntimeError('organ_crop=off 的前端收到 crop —— dataset 接错线了')
        self.stats['crop_ok'] += 1
        key = (sid, 'c')
        m = self._mem.get(key)
        if m is not None:
            self._mem.move_to_end(key)
            self.stats['hit_mem'] += 1
            out, lay = self._masked(m, crop)
            return (*self._process(out), lay)
        p = self.path(split, sid)
        if not os.path.isfile(p):
            slices, idx = self._from_volume(sid, split, self.mreq,
                                            allow_fewer=True, vol=vol)
            if self.write:
                self._save(p, slices, idx)
            self._put(key, slices, idx)
            self.stats['load'] += 1
            out, lay = self._masked((slices, idx), crop)
            return (*self._process(out), lay)
        with np.load(p) as z:
            sl, idx = z['slices'], z['indices']
            if sl.shape[0] != len(idx) or sl.shape[0] > self.mreq:
                raise RuntimeError(f'{p} 里是 {sl.shape[0]} 片（indices {len(idx)} 个），'
                                   f'本次 mreq={self.mreq} —— 缓存名带 Mreq'
                                   '（`<Mreq>c_<sid>.npz`）但仍撞上，说明目录串了')
            self._put(key, sl, idx)
            self.stats['hit_disk'] += 1
        out, lay = self._masked((sl, idx), crop)
        return (*self._process(out), lay)

    def _masked(self, m: tuple, crop: tuple) -> tuple:
        """缓存的 (slices, idx) + crop -> (送 processor 的片堆, 选中片的层号)。

        crop 前 6 位是 bbox，第 7 位（仅 mask 臂）是逐层器官体素剖面。`z` 只按 bbox 的
        z 范围选层、平面保持全幅；`bbox` 再加平面裁；`mask` 按剖面选层、平面全幅。
        """
        slices, idx = m
        z0, z1, y0, y1, x0, x1 = (int(v) for v in crop[:6])
        zprof = crop[6] if len(crop) > 6 else None
        if self.organ_crop == 'mask':
            if zprof is None:
                self.stats['mask_noprof'] += 1
                sel = crop_select(idx, z0, z1, self.n_slices, self.stats)
            else:
                sel = mask_select(idx, zprof, z0, z1, self.n_slices, self.stats)
        else:
            sel = crop_select(idx, z0, z1, self.n_slices, self.stats)
        out = slices[sel]
        if self.organ_crop == 'bbox':
            out = out[:, y0:y1 + 1, x0:x1 + 1]
        return out, np.asarray(idx)[sel]

    def _from_volume(self, sid: str, split: str, n_want: int,
                     allow_fewer: bool = False, vol: str | None = None):
        """现读体积出 n_want 片。`allow_fewer`：层数不足时取满全层 —— 仅裁剪臂缓存用；
        off 臂不松（D < n_slices 照旧抛错，那是序列占位数会裂的硬错）。

        `vol`：数据集记录自带的图像路径。按 sid 拼出的默认路径不存在（外源 sid
        不在 iid 下）时才用它 —— IID 侧两条路径同一文件，不受影响。
        """
        p = config.image_path(sid)
        if not os.path.isfile(p) and vol:
            p = config.image_path(sid, vol)
        img, _meta = volume.load_and_prepare(p)
        arr = img[0].numpy()
        if any(int(s) > 1000 for s in arr.shape):
            raise RuntimeError(f'{sid} padded 后 {arr.shape} 有维 >1000，上游跳过口径')
        if allow_fewer:
            n_want = min(n_want, arr.shape[0])
        idx = pick_indices(arr.shape[0], n_want)
        return (np.clip(arr[idx], 0.0, 1.0) * 255.0).round().astype(np.uint8), idx

    def _save(self, dst: str, slices: np.ndarray, idx: np.ndarray) -> None:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix='.tmp_', suffix='.npz')
        os.close(fd)
        try:
            np.savez(tmp, slices=slices, indices=idx)
            os.replace(tmp, dst)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _put(self, key: tuple, slices: np.ndarray, idx: np.ndarray) -> None:
        self._mem[key] = (slices, idx)
        while len(self._mem) > self.lru:
            self._mem.popitem(last=False)

    def _proc_cached(self, key: tuple, slices) -> tuple[torch.Tensor, torch.Tensor]:
        """`_process` 的备忘（`proc_lru>0` 才开；缺省 0 = 逐位是老行为）。

        为什么值得记：`_process` 是 raw uint8 切片的**纯函数**，同一 key 的产物逐位相同
        （`analysis/check_process_determinism.py` 核过 5 卷 × OMP 1/2/4/8/16/不设 六档的
        sha256 全同）。而它很贵 —— 32 片 345×410 实测 0.70 s（OMP=1）/ 4.2 s（不设，
        40 线程在小张量上互踩），产物 18304×1536 fp32 = 112.5 MB。上面那层 `_mem` 只
        省了盘上 npz 的读，**命中了照样重跑 processor**（275 行那条），所以整卷 eval 的
        墙钟全砸在这儿：GPU util 0%、进程 CPU 1875%。

        同卷的题是**连着出**的 —— test.json 56825 题 / 5080 卷 = 每卷 11.19 题且卷内连续
        （实测 hit_mem 91%）；`StudyShuffledSampler` 是 study 级洗牌、卷内保持原序
        （dataset.py 那句「记录与嵌入只读一次」），SFT/align 臂实测同样 90.9%。所以记下
        processed 产物能把整卷 eval 的 `_process` 调用从 56825 压到 5080。
        `GroupStratifiedSampler`（RACD/GRPO）按器官组重排、打散了卷内连续性，实测只有
        40.8% —— 收益按采样器而异，不是统一的 11×。

        🔴 只在 off 口径（`crop=None`）挂这层：裁剪臂的 `_process` 吃的是**按题算出来的
        层子集**，同卷不同题产物不同，key 撞了就是错的数。`_pixels_crop` 不走这里。
        🔴 返回的是**同一个张量对象**，同卷多行共享。当前下游只读：collate 的
        `torch.cat`（dataset.py）与 `model._native_embeds` 前的 `.to(x.device, x.dtype)`
        （fp32→bf16 必拷贝）。RACD/GRPO 那条还要多一层：`MultiViewDataset.__getitem__`
        只调一次 `base[i]`、再 `d = dict(full)` 摊成 M 个视图（views.py），所以 M 个视图
        的 `pixel_values` 也是同一个对象 —— 它 clone 了 `has_feat`/`organ_box` 再写，
        唯独像素不动。任何新加的 in-place 都会静默污染同卷其它题**和同题其它视图**，而
        loss 曲线看起来完全正常 —— 改这条路径前先想这件事。
        """
        if self.proc_lru <= 0:
            return self._process(slices)
        m = self._pmem.get(key)
        if m is not None:
            self._pmem.move_to_end(key)
            self.stats['hit_proc'] += 1
            return m
        out = self._process(slices)
        self._pmem[key] = out
        while len(self._pmem) > self.proc_lru:
            self._pmem.popitem(last=False)
        self.stats['miss_proc'] += 1
        return out

    def _process(self, slices) -> tuple[torch.Tensor, torch.Tensor]:
        """uint8 切片 -> 宿主的 `pixel_values` + `image_grid_thw`（逐样本，不带文本）。

        🔴 走的是 `proc.image_processor` 而**不是** `proc(...)`：完整的 processor 要求
        `text`（它要在文本里把 `image_token` 展开成占位符串，实测 `text=None` 直接
        `TypeError: argument of type 'NoneType' is not iterable`）。我们不扩 —— 序列里的
        占位符由 `pack.build_example` 手工拼（这些特殊 id 过 tokenizer 往返会被丢，见
        pack.py 的 🔴 段）。像素与 grid 这两样，`image_processor` 给的和 `proc` 给的是
        同一个函数（`analysis/smoke_q35_b.py` 里拿 processor 的规范展开对拍过占位数）。
        """
        a = np.asarray(slices)
        imgs = [Image.fromarray(a[i]).convert('RGB') for i in range(a.shape[0])]
        out = self.proc.image_processor(images=imgs, return_tensors='pt')
        return out['pixel_values'], out['image_grid_thw']

    def describe(self) -> dict:
        return {'n_slices': self.n_slices, 'root': self.root, 'write': self.write,
                'lru': self.lru, 'proc_lru': self.proc_lru,
                'host': os.path.basename(self.host_path.rstrip('/')),
                'organ_crop': self.organ_crop, 'crop_margin': self.crop_margin,
                'crop_cache_n': self.crop_cache_n, 'mreq': self.mreq,
                **self.info, 'stats': dict(self.stats)}
