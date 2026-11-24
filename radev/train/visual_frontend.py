"""视觉前端 —— 冻结的 RADAR/RADAR+ 塔在训练进程里按需出特征（「不预提取」那条路）。

为什么要有这个模块（2026-10-12 的决定：「不用预先提取了，冻结视觉 encoder，作为一个 vlm 来跑」）：
stage 1 那条独立预提取流水线（`cache.py`）先把 25,484 例编码成 npz，训练只读盘。现算把
**同样的 FLOPs 挪进训练循环**，所以这里必须把代价说清楚并管住：

    实测（out_radar_emb/logs/shard_*.log，4 片各 17.3 h）：0.10 例/s ≈ 9.8 s/例，
    其中滑窗只有 4.25 窗/例（GPU ~2 s），其余 ~8 s 是 nii.gz 解码 + trilinear resize。
    ⇒ 单卡一个 epoch 现算 ≈ 15,306 × 9.8 s ≈ 42 h；重启/下一轮从头再来。

因此前端是 **三级取数**，不是纯现算：

    RAM LRU  ->  盘上 npz（write-through 的产物，读回 ~ms）  ->  现算并落盘

也就是「lazy 版 stage 1」：第一条命令就把预提取、续跑、缓存对账合成一件事，npz 的列语义
与 `cache.py` 逐字相同（同一个 `save_npz`），所以 pretrain 那支已经跑完的 25,484 个 npz
可以直接当第 2 级用，radar_ft 这支则边训边长出来。

三条不许含糊的点：
1. **编码器身份仍由 `radar_encoder.ENCODER_IDENTITIES` 声明**（a17）：这里只接
   `identity`，不认路径的形状。`radar_ft` 见过 the source CT training cohort（≈ 我们的 train+val），
   它的特征列可用、它的逐例概率不能当第三方佐证。
2. **本模块只在主进程用**（`num_workers=0`）：塔在 CUDA 上，DataLoader 的 worker 是 fork
   出来的进程，CUDA context 不可继承。`_feats()` 里当场检测 worker 并拒绝，而不是让
   它在 8 个 worker 里各崩一次。
3. **上游的 >1000 跳过规则保留**（`infer_source_testset.py`：padded 任一维 >1000 就不做
   滑窗）。这类 study 不静默给零向量 —— `has_feat` 全 False 进 projector 的可学 `missing`
   槽，并计数写进 run info（数量与身份都是要报的东西，不是实现细节）。
"""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from .. import config
from . import radar_encoder as RE
from . import volume

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(REPO, 'feature_cache')
N_ROWS_ORGANS = RE.N_ORGANS


class SkipStudy(Exception):
    """这例按上游口径不产特征（不是错误，是数据边界）。"""


def read_npz_rows(p: str, *, want_whole: bool, who: str = 'cache') -> tuple[np.ndarray, np.ndarray]:
    """npz -> (feats[n_rows,256] float16, has[n_rows] bool)。列语义的唯一读法。

    `n_rows = 36 (+1 整图)`；第 37 行由 `whole_feat`/`has_whole` 拼出来，所以「这支有没有
    整图列」是身份问题，不是形状问题 —— 拿 radar_ft 的口径去读 pretrain 缓存会当场报错，
    而不是静默补一行 0（那等于喂给模型一个恒为 0 的 token 而没有任何提示）。
    """
    with np.load(p) as z:
        feats, has = z['feats'], z['has_feat']
        if feats.shape != (N_ROWS_ORGANS, 256) or has.shape != (N_ROWS_ORGANS,):
            raise RuntimeError(f'{p} 的形状 {feats.shape}/{has.shape} 与身份 {who} '
                               '的列语义不符（a15）')
        feats = feats.astype(np.float16)
        has = has.astype(bool)
        if not want_whole:
            return feats, has
        if 'whole_feat' not in z.files:
            raise RuntimeError(f'{p} 没有 whole_feat 列，但身份 {who} 要第 37 槽 '
                               '—— 这份缓存不是这一支产的（或产它时没开整图）')
        w = z['whole_feat'].astype(np.float16)
        hw = bool(np.asarray(z['has_whole']).item())
        return (np.vstack([feats, w[None, :]]),
                np.concatenate([has, np.asarray([hw], bool)]))


def read_npz_box(p: str) -> np.ndarray | None:
    """npz -> organ_box int16[36,6]（坐标序 z0,z1,y0,y1,x0,x1，上下界均含），缺列 → None。

    缺列是可预期状态（pretrain 老缓存没这一列、不重刷），所以 None 不是错误；但列在
    而形状/类型不对 = 有东西写坏了，当场报错。None 的下游语义：dataset 侧 crop=None，
    逐题回退 off 臂口径并计数（见 slice_stats.json）。
    """
    with np.load(p) as z:
        if 'organ_box' not in z.files:
            return None
        box = z['organ_box']
        if box.shape != (N_ROWS_ORGANS, 6) or box.dtype != np.int16:
            raise RuntimeError(f'{p} 的 organ_box {box.shape}/{box.dtype} != '
                               f'({N_ROWS_ORGANS}, 6)/int16')
        return box


def read_npz_zprof(p: str) -> np.ndarray | None:
    """npz -> organ_zprof int32[36,D]（逐层器官体素计数），缺列 → None。

    缺列是老缓存（本列 2026-10-19 才加）的可预期状态；列在而形状/类型不对 = 写坏了。
    None 的下游语义：mask 选层臂逐题回退 z 口径（bbox 的 z 范围、平面全幅），
    dataset 侧计 mask_noprof。
    """
    with np.load(p) as z:
        if 'organ_zprof' not in z.files:
            return None
        a = z['organ_zprof']
        if a.ndim != 2 or a.shape[0] != N_ROWS_ORGANS or a.dtype != np.int32:
            raise RuntimeError(f'{p} 的 organ_zprof {a.shape}/{a.dtype} != '
                               f'({N_ROWS_ORGANS}, D)/int32')
        return a


def save_npz(dst: str, res: dict, geom: dict | None = None) -> None:
    """npz 列的唯一写法（`cache.py` 与本模块共用 —— 列语义不该有两份真值）。

    res 键：organ_feats / organ_sizes / probs / sag_voxels / n_windows / stitched_mask /
    whole_feat(可缺)

    `geom`（A2 用）：把**几何四列**换成从别处搬来的现成列 ——
    `{'organ_box': int16[36,6], 'organ_zprof': int32[36,D], 'sizes': int32[36],
    'sag_voxels': float}`。缺哪个键就用 res 现算的那个（不默认补 0/-1）。
    为什么要有这个口子：`organ_box`/`organ_zprof` 的坐标系必须与**切片缓存**那一卷
    一致（`slice_frontend.py:10`：切片取自 `load_and_prepare` 的非零区 crop + pad），
    而 A2 的 mask 来自 `load_and_prepare_whole` 定形的另一卷。特征要新的，几何不能新
    —— 否则证据 token 拿别人的 box 框自己的层，形状合法、读数全假。
    """
    n = RE.N_ORGANS
    feats = np.zeros((n, 256), np.float16)
    has = np.zeros(n, bool)
    sizes = np.zeros(n, np.int32)
    for oi, organ in enumerate(RE.ORGANS):
        t = res['organ_feats'].get(organ)
        if t is not None:
            a = t.numpy().reshape(-1)
            if a.shape != (256,):
                raise RuntimeError(f'器官 {organ} 的向量形状 {a.shape} != (256,)')
            feats[oi] = a.astype(np.float16)
            has[oi] = True
        sizes[oi] = int(res['organ_sizes'].get(organ, 0))

    w = res.get('whole_feat')
    if w is None:
        whole = np.zeros(256, np.float16)
        has_whole = False
    else:
        whole = w.numpy().reshape(-1)
        if whole.shape != (256,):
            raise RuntimeError(f'整图向量形状 {whole.shape} != (256,)')
        if not np.isfinite(whole).all():
            raise RuntimeError('整图向量非有限（NaN/Inf）—— 归一化 0/0，见 volume 注释')
        has_whole = True
        whole = whole.astype(np.float16)

    keys = [it for it in RE.PROB_ITEMS if it in res['probs']]
    probs = np.asarray([res['probs'][k] for k in keys], np.float32)
    if geom is not None and 'organ_box' in geom:
        organ_box = np.asarray(geom['organ_box'])
    else:
        organ_box = RE.organ_boxes(res['stitched_mask'])
    if geom is not None and 'organ_zprof' in geom:
        organ_zprof = np.asarray(geom['organ_zprof'])
    else:
        organ_zprof = RE.organ_zprofiles(res['stitched_mask'])
    if geom is not None and 'sizes' in geom:
        sizes = np.asarray(geom['sizes'], np.int32)
    sag = float(geom['sag_voxels']) if (geom is not None and 'sag_voxels' in geom) \
        else float(res['sag_voxels'])
    d = os.path.dirname(dst)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix='.tmp_', suffix='.npz')
    os.close(fd)
    try:
        np.savez(tmp, feats=feats, has_feat=has, sizes=sizes,
                 whole_feat=whole, has_whole=np.asarray(has_whole),
                 probs=probs, prob_keys=np.asarray(keys),
                 sag_voxels=np.float32(sag),
                 n_windows=np.int32(res['n_windows']),
                 organ_box=organ_box, organ_zprof=organ_zprof)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def encode_study(model: torch.nn.Module, img_path: str, device: str, *,
                 want_whole: bool, pad_func, want_probs: bool = False,
                 text_emb: dict | None = None, infer_form: str = 'window') -> dict:
    """一例 -> res（`save_npz` 的入参形状）。概率一律不算，除非显式给了文本嵌入。

    `infer_form`（A2）：`window` = 上游官方口径（非零区 crop + 滑窗），`nowin` = 只用
    权重不用它的推理形式（整图支路那卷定形输入，单次前向，逐器官池化）。两者产的 res
    同构，但**不是同一个函数**，所以形态由编码器身份带进来（`RE.infer_form_of`），
    不是由调用方随手选。

    两遍解码是故意的（仅 `window` 形态）：器官支路与整图支路在 resize 之后就分叉
    （窗宽、crop 方式都不同），而器官那一路会**原地**改体素（clip + 归一化写在同一块
    张量上）。共享那张大张量要么得到被改过的一半，要么峰值内存翻倍 —— 一遍 1.5 GB 的
    重新解码比这两个都便宜。`nowin` 形态下这两支**同源**（都用 `load_and_prepare_whole`
    的定形卷），所以只解一次；共享是安全的：两个前向都只是读输入。
    """
    if infer_form == 'window':
        image, _meta = volume.load_and_prepare(img_path)
        if any(int(s) > 1000 for s in image.shape[1:]):
            raise SkipStudy(f'padded {tuple(image.shape[1:])} 有维 >1000（上游跳过口径）')
        res = RE.infer_volume(
            model, image[None], RE.TEST_ITEMS if want_probs else [],
            text_emb or {}, device=device, pad_func=pad_func, compute_organs=RE.ORGANS)
        del image, _meta
        res['whole_feat'] = None
        if want_whole:
            wimg, _ = volume.load_and_prepare_whole(img_path)
            res['whole_feat'] = RE.infer_whole(model, wimg[None], device=device)
        return res
    if infer_form != 'nowin':
        raise ValueError(f'infer_form={infer_form!r} 不在 {RE.INFER_FORMS}')
    wimg, _ = volume.load_and_prepare_whole(img_path)
    res = RE.infer_volume_nowin(
        model, wimg[None], RE.TEST_ITEMS if want_probs else [],
        text_emb or {}, device=device, compute_organs=RE.ORGANS)
    res['whole_feat'] = RE.infer_whole(model, wimg[None], device=device) if want_whole else None
    return res


class VisualFrontend:
    """`sid -> (feats[n_tok,256] float32, has_feat[n_tok] bool)`，三级取数 + 计数。

    n_tok = 36（器官，按 `RE.ORGANS` 定序）+ 1（整图，仅 `has_whole` 的身份有）。
    第 37 槽就是整图 token：它排在器官之后，projector 给它自己的槽位 id 嵌入。
    """

    def __init__(self, identity: str = 'radar_ft', *, ckpt: str | None = None,
                 device: str = 'cuda', root: str = DEFAULT_ROOT, write: bool = True,
                 lru: int = 2048, probs: bool = False,
                 text_emb_path: str | None = None) -> None:
        if identity not in RE.ENCODER_IDENTITIES:
            raise ValueError(f'未登记的编码器身份 {identity}；可选 {sorted(RE.ENCODER_IDENTITIES)}')
        self.ident = RE.ENCODER_IDENTITIES[identity]
        self.identity = identity
        self.infer_form = self.ident.get('infer_form', RE.DEFAULT_INFER_FORM)
        self.geom_frame = 'window' if self.infer_form == 'window' else 'nowin'
        self.device = device
        self.root = root
        self.write = bool(write)
        self.want_whole = bool(self.ident['has_whole'])
        self.n_rows = N_ROWS_ORGANS + (1 if self.want_whole else 0)
        self.lru = int(lru)
        self._mem: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.stats = {'hit_mem': 0, 'hit_disk': 0, 'encode': 0, 'skip_gt1000': 0,
                      'encode_failed': 0}

        cls = getattr(RE, self.ident['cls'])
        ck = ckpt or os.path.join(
            os.environ.get('RADAR_CKPT_ROOT') or os.path.join(config.ROOT, 'radar_ckpt'),
            self.ident['basename'])
        self.model, self.info = cls.from_pretrained(ck, identity=identity, device=device)
        self.pad_func = RE.default_pad_func()

        self.probs = bool(probs)
        self.text_emb: dict[str, torch.Tensor] = {}
        if self.probs:
            if self.ident['text_lang'] != 'zh':
                raise ValueError(f'身份 {identity} 的文本塔是 {self.ident["text_lang"]}，'
                                 '而现算概率目前只有中文嵌入（#60 之前不许开 --probs）')
            p = text_emb_path or os.path.join(
                os.environ.get('RADAR_CKPT_ROOT') or os.path.join(config.ROOT, 'radar_ckpt'),
                'infer_text_embedding.pt')
            td = torch.load(p, map_location='cpu', weights_only=True)
            self.text_emb = {k: v.to(device) for k, v in td.items()}
            self.info['text_emb_sha256'] = RE.sha256_of(p)

    def path(self, split: str, sid: str) -> str:
        return os.path.join(self.root, self.info['ckpt_sha256'][:12], split, f'{sid}.npz')

    def feats(self, sid: str, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        info = torch.utils.data.get_worker_info()
        if info is not None:
            raise RuntimeError(
                f'VisualFrontend 不能在 DataLoader worker（num_workers={info.num_workers}）里用：'
                '塔在 CUDA 上，fork 出去的进程没有可用的 CUDA context。'
                '把 dataloader 的 num_workers 设回 0（本仓库的训练循环就是这个口径）')
        m = self._mem.get(sid)
        if m is not None:
            self._mem.move_to_end(sid)
            self.stats['hit_mem'] += 1
            return torch.from_numpy(m[0].astype(np.float32)), torch.from_numpy(m[1].copy())
        p = self.path(split, sid)
        if os.path.isfile(p):
            f = read_npz_rows(p, want_whole=self.want_whole, who=self.identity)
            self._put(sid, f[0], f[1])
            self.stats['hit_disk'] += 1
            return torch.from_numpy(f[0].astype(np.float32)), torch.from_numpy(f[1].copy())
        return self._encode(sid, split, p)

    def _encode(self, sid: str, split: str, p: str):
        img = config.image_path(sid)
        try:
            res = encode_study(self.model, img, self.device, want_whole=self.want_whole,
                               pad_func=self.pad_func, want_probs=self.probs,
                               text_emb=self.text_emb, infer_form=self.infer_form)
        except SkipStudy as e:
            self.stats['skip_gt1000'] += 1
            f = np.zeros((self.n_rows, 256), np.float16)
            h = np.zeros(self.n_rows, bool)
            self._put(sid, f, h)
            print(f'[visual] ⚠️ {sid}: {e} —— has_feat 全 False（可学 missing 槽）',
                  flush=True)
            return torch.from_numpy(f.astype(np.float32)), torch.from_numpy(h)
        except Exception as e:
            self.stats['encode_failed'] += 1
            raise RuntimeError(f'{sid} 现算特征失败：{type(e).__name__}: {e}') from e
        self.stats['encode'] += 1
        if self.write:
            save_npz(p, res)
        f, h = self._rows_from_res(res)
        self._put(sid, f, h)
        return torch.from_numpy(f.astype(np.float32)), torch.from_numpy(h)

    def _rows_from_res(self, res: dict) -> tuple[np.ndarray, np.ndarray]:
        f = np.zeros((self.n_rows, 256), np.float16)
        h = np.zeros(self.n_rows, bool)
        for oi, organ in enumerate(RE.ORGANS):
            t = res['organ_feats'].get(organ)
            if t is not None:
                f[oi] = t.numpy().reshape(-1).astype(np.float16)
                h[oi] = True
        if self.want_whole:
            w = res.get('whole_feat')
            if w is not None:
                f[-1] = w.numpy().reshape(-1).astype(np.float16)
                h[-1] = True
        return f, h

    def _put(self, sid, f: np.ndarray, h: np.ndarray) -> None:
        self._mem[sid] = (f, h)
        while len(self._mem) > self.lru:
            self._mem.popitem(last=False)

    def describe(self) -> dict[str, Any]:
        return {
            'visual_mode': 'live',
            'encoder_identity': self.identity,
            'infer_form': self.infer_form,
            'geom_frame': self.geom_frame,
            'ckpt_sha256': self.info['ckpt_sha256'],
            'n_tensors_visual': self.info['n_tensors_visual'],
            'has_whole': self.want_whole,
            'n_rows': self.n_rows,
            'saw_radar_train': self.info['saw_radar_train'],
            'published_csv_anchor': self.info['published_csv_anchor'],
            'text_lang': self.info['text_lang'],
            'root': os.path.abspath(self.root),
            'write_through': self.write,
            'lru': self.lru,
            'probs': self.probs,
        }
