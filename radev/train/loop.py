"""训练 / 评估 / 地板：`python -m radev train {train,eval,floors} …`（plan 步骤 5、6）。

三条子命令共用同一套构建路径：

* `train` —— bf16 前向 + 梯度累积 + AdamW + 检查点续跑 + `--limit` 干跑。
  一序列一题、study 级 shuffle（`StudyShuffledSampler`）；loss 走 `JevModel.forward()`
  （teacher forcing，见 model.py 的两条路径说明）。
* `eval` —— 载回检查点，**报数只走 `score_exact()`**（逐候选前缀的精确分数），
  metrics.py 算逐题指标，并按 floor.py 的 ①② 两地板做 McNemar 判定。
* `floors` —— 只算 ①② 两级地板（零训练），落 `floors.json`。

run_info.json（plan §6 第 9 条）记：rules 白名单、缓存目录与编码器身份、LLM 路径与
廉价身份（config.json 的 sha256 + 权重字节数）、projector 形状、feats 模式（**zeros 是
冒烟模式，写明「不许报数」**）、`--feats live` 时视觉前端的身份（a17：encoder_identity /
ckpt sha256 / 张量数 / 是否见过 the source training cohort）、n_vis、`--limit`、命令行。
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import hashlib
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import config
from . import floor as FL
from . import metrics as M
from . import pack
from . import radar_encoder as RE
from .dataset import (DEFAULT_NEWQ_DIR, FINDING_RULES, JevDataset,
                      StudyShuffledSampler, resolve_cache, select_vis_tokens)
from .model import JevModel
from .radar_encoder import N_ORGANS
from .visual_frontend import DEFAULT_ROOT as DEFAULT_VISUAL_ROOT


DEFAULT_ORGAN_MAP = os.path.join(config.ROOT, 'out_inventory', 'organ_crop_map.tsv')


def _load_organ_map(path, *, need=True):
    """`organ_crop_map.tsv` -> ({题名: [器官名]}, sha256, 行数)。空组 = `[]`（要进 dict）。

    `need=False`（off 臂）：文件不在返回 (None, None, 0)，不拦路 —— off 臂从不查表。
    表内器官名必须 ∈ `radar_encoder.ORGANS`：名字在别处的表会静默错位（查不到的行
    变成 no_box，而不是报错），所以在这里当场拒。
    """
    from .radar_encoder import ORGANS
    if not os.path.isfile(path):
        if need:
            raise SystemExit(f'缺器官表 {path} —— 先生成它：`<jjt_mllm>/bin/python '
                             'analysis/organ_crop_map.py`（幂等可重跑），再经 --organ-map 指定')
        return None, None, 0
    q2o = {}
    with open(path, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        if head[:3] != ['question', 'group', 'organs']:
            raise SystemExit(f'{path} 表头 {head[:3]} != [question, group, organs] —— '
                             '不是这份脚本的产物？')
        for ln, line in enumerate(f, 2):
            d = line.rstrip('\n').split('\t')
            q = d[0]
            orgs = d[2].split() if len(d) > 2 else []
            bad = [o for o in orgs if o not in ORGANS]
            if bad:
                raise SystemExit(f'{path}:{ln} 未知器官 {bad}（对照 radar_encoder.ORGANS）—— '
                                 '改 analysis/organ_crop_map.py 的表重跑，别手改本表')
            if q in q2o:
                raise SystemExit(f'{path}:{ln} 题名 {q} 重复')
            q2o[q] = orgs
    with open(path, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return q2o, sha, len(q2o)


def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


def _write_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def _log_jsonl(path, obj):
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')


def _load_tokenizer(args):
    from transformers import AutoTokenizer
    cands = [args.tok] if args.tok else []
    if not cands and getattr(args, 'llm', None):
        cands.append(args.llm)
    cands += [os.path.join(config.ROOT, 'pretrained', n)
              for n in ('Qwen3.5-0.8B', 'Qwen3.5-2B', 'Qwen3.5-4B', 'Qwen3.5-9B')]
    for d in cands:
        if d and os.path.isdir(d):
            tok = AutoTokenizer.from_pretrained(d)
            return tok, d
    raise SystemExit(f'找不到 tokenizer：{cands}（用 --tok 指一个 Qwen3.5 系目录）')


def _llm_identity(path):
    """廉价身份：config.json 的 sha256 + 权重字节数（不重算 3.4 GB 的哈希）。"""
    out = {'path': path, 'exists': os.path.isdir(path)}
    if not out['exists']:
        return out
    cfg = os.path.join(path, 'config.json')
    if os.path.isfile(cfg):
        with open(cfg, 'rb') as f:
            out['config_sha256'] = hashlib.sha256(f.read()).hexdigest()[:16]
    tot = 0
    for name in sorted(os.listdir(path)):
        p = os.path.join(path, name)
        if os.path.isfile(p) and (name.endswith('.safetensors') or name.endswith('.bin')):
            tot += os.path.getsize(p)
    out['weight_bytes'] = tot
    return out


def _dev(args):
    """tiny 强制 CPU；否则有卡就用卡。视觉前端与模型必须同一个 dev（塔的权重落在谁上）。"""
    return 'cuda' if torch.cuda.is_available() and not args.tiny else 'cpu'


def _build_frontend(args, dev):
    """`--feats live` -> 一个进程一份冻结塔（`VisualFrontend`）；其余模式 None。

    身份（`--encoder`）在这里只声明一次，dataset / projector / run_info 都从它取值 ——
    「第 37 槽存在吗」只能是身份的属性，不能是三处各自的默认值（a15/a17）。
    """
    if args.feats != 'live':
        return None
    if args.mode == 'state-only':
        print('[visual] mode=state-only：序列里 0 个视觉占位符 —— 不建前端（载入 1.5 GB '
              '权重换 0 个 token 没有意义），feats 的语义记在 run_info 里', flush=True)
        return None
    if args.tiny:
        raise SystemExit('--tiny 与 --feats live 不合：冒烟请配 --feats zeros'
                         '（现算要真塔，随机塔出来的特征没有意义）')
    from .visual_frontend import VisualFrontend
    return VisualFrontend(args.encoder, ckpt=args.visual_ckpt, device=dev,
                          root=args.visual_root, write=not args.no_write_through,
                          lru=args.lru)


def _build_sfrontend(args):
    """B 支的第二条通路：`--slices N>0` 建一个 `SliceFrontend`（读宿主自己的 processor），
    `0` = 只走 RADAR 一条通道（A 支）。

    宿主路径就是 `--llm`：processor 的 patch/merge/mean-std 与权重同源，指两个目录会得到
    「grid 与占位数不一致」的序列 —— 所以这里不开第二个入口参数。
    """
    if not args.slices:
        return None
    if not args.llm:
        raise SystemExit('--slices 要 --llm 指一个原生多模态宿主目录'
                         '（processor 与权重必须同目录）')
    from .slice_frontend import SliceFrontend
    return SliceFrontend(args.llm, args.slices, root=args.slice_root,
                         write=not args.no_write_through, lru=args.slice_lru,
                         proc_lru=args.slice_proc_lru,
                         organ_crop=args.organ_crop, crop_margin=args.crop_margin,
                         crop_cache_n=args.crop_cache_n)


def _has_whole(args, frontend):
    """projector 要不要开第 37 槽：live 看身份，cache 看 meta，zeros 没有真值来源。"""
    if frontend is not None:
        return bool(frontend.want_whole)
    if args.feats == 'cache':
        _d, meta = resolve_cache(args.cache_root, args.cache_sha12)
        return bool(meta.get('has_whole'))
    return False


def _proj_geom(args, frontend):
    """(projector 的 n_organs, has_whole, 序列里的视觉 token 数)。

    `_has_whole` 给的是**缓存里有几行**，`--vis-tokens` 决定**其中哪几行进序列**（A3/A4）：
    去整图 ⇒ projector 建成 36 槽（`has_whole=False`），只整图 ⇒ 建成 1 槽
    （`n_organs=0, has_whole=True`）。model 与 dataset 必须各自调用后得到同一个数 ——
    两边都走 `dataset.select_vis_tokens` 这一份实现，`_check_rows` 才是事后核对而不是
    唯一防线。`feats=none`（纯 2D）没有特征行，原样返回（此时 n_vis 由 `_n_vis` 归 0）。
    """
    n = int(args.n_organs or N_ORGANS)
    hw = _has_whole(args, frontend)
    if args.feats == 'none' or args.mode == 'state-only':
        return n, hw, 0
    rows, pn, ph = select_vis_tokens(n + (1 if hw else 0), n,
                                     str(getattr(args, 'vis_tokens', 'all') or 'all'))
    return pn, ph, len(rows)


def _build_model(args, tok, frontend=None):
    if getattr(args, 'organ_route_visual', False) or getattr(
            args, 'organ_route_perorgan', False):
        args.organ_route = True
    pn, phw, _nvis = _proj_geom(args, frontend)
    vt = str(getattr(args, 'vis_tokens', 'all') or 'all')
    oi = str(getattr(args, 'organ_index', 'on'))
    if oi not in ('on', 'off'):
        raise SystemExit(f"--organ-index={oi} 只能是 on/off（off = A1：槽位身份嵌入换成"
                         '一个共享向量，配套 --slot-perm 打乱行序）')
    index = (oi != 'off')
    if not index and bool(getattr(args, 'pure2d', False)):
        raise SystemExit('--organ-index off 与 --pure2d 不合：纯 2D 臂的 projector 不进计算图，'
                         '索引对它本来就不存在')
    fp = int(getattr(args, 'fusion_pool', 0) or 0)
    o2d = bool(getattr(args, 'organ2d', False))
    oev = bool(getattr(args, 'organ_evidence', False))
    evnb = bool(getattr(args, 'evidence_no_box', False))
    rt = bool(getattr(args, 'organ_route', False))
    rv = bool(getattr(args, 'organ_route_visual', False))
    rp = bool(getattr(args, 'organ_route_perorgan', False))
    oa = bool(getattr(args, 'organ_acquire', False))
    if oa and not oev:
        raise SystemExit('--organ-acquire 要 --organ-evidence：获取动作掩的是证据槽，'
                         '没有证据臂就没有可获取的东西（JevModel.__init__ 也有这条不变量，'
                         '这里在读宿主权重之前先拦）')
    if vt == 'whole':
        bad = [n for n, v in ((' --organ-evidence', oev), ('--organ2d', o2d),
                              ('--organ-route', rt or rv or rp),
                              ('--fusion-pool', bool(fp)), ('--fuse', bool(args.fuse)),
                              ('--evidence-no-box', evnb)) if v]
        if int(args.slices or 0) > 0:
            bad.append('--slices>0')
        if str(getattr(args, 'organ_slots', 'on')) == 'off':
            bad.append('--organ-slots off')
        if bad:
            raise SystemExit('--vis-tokens whole（A3：序列里只有那一个整图 token）与 '
                             + ' / '.join(bad) + ' 不合 —— 留着它们「一个 token」就是假的')
    elif vt not in ('all', 'organs'):
        raise SystemExit(f'--vis-tokens={vt} 只能是 all / organs（A4）/ whole（A3）')
    if vt == 'organs' and str(getattr(args, 'organ_slots', 'on')) == 'off':
        raise SystemExit('--vis-tokens organs 与 --organ-slots off 不合：后者序列里一个 3D 槽'
                         '都没有（3D 只作融合 K/V），选哪几行无从谈起')
    if evnb and not oev:
        raise SystemExit('--evidence-no-box 要 --organ-evidence：它放宽的正是证据 token 的 '
                         'K/V 归属，没有证据臂就没有可放宽的东西（无归属的器官池化是 '
                         '--fusion-pool，另一条臂）')
    if args.tiny:
        model = JevModel.tiny(tok, n_organs=pn, has_whole=phw,
                              vl=bool(args.slices), fuse=bool(args.fuse), pool=fp,
                              organ2d=o2d, evidence=oev, route=rt, route_visual=rv,
                              route_perorgan=rp, ev_nobox=evnb, index=index, acquire=oa)
        ident = {'tiny': True}
        if args.llm:
            ident['processor_dir'] = args.llm
    else:
        if not args.llm:
            raise SystemExit('真训练必须给 --llm <模型目录>'
                             '（Qwen3.5-*；A/B 两支同一宿主，A 支只是不喂切片；'
                             '或 --tiny 走 CPU 冒烟）')
        model = JevModel.from_pretrained(args.llm, n_organs=pn,
                                         dtype=args.dtype, freeze_llm=args.freeze_llm,
                                         attn_implementation=args.attn or None,
                                         has_whole=phw, fuse=bool(args.fuse),
                                         pool=fp, organ2d=o2d, evidence=oev, route=rt,
                                         route_visual=rv, route_perorgan=rp,
                                         ev_nobox=evnb, index=index, acquire=oa,
                                         pad_id=tok.pad_token_id)
        ident = _llm_identity(args.llm)
    if getattr(args, 'pure2d', False):
        model.projector.requires_grad_(False)
    return model, ident


def _n_vis(args):
    if getattr(args, 'pure2d', False):
        return 0
    if args.mode == 'state-only':
        return 0
    if str(getattr(args, 'organ_slots', 'on')) == 'off':
        return 0
    if args.mode == 'full':
        return None
    raise SystemExit(f'--mode={args.mode} 只能 full / state-only（radar-cosine 走 floors 子命令）')


def _json_path(args, prompt_only):
    """训练/评估各自的数据面。`None` = 退回内部题面（keep TSV + out_v3）。"""
    return getattr(args, 'eval_json' if prompt_only else 'train_json', None)


def _check_json_pair(args, who):
    """训练与评估的 JSON 必须成对，且不能是同一个文件。

    只给一半 = 训练用导出题面、评估用内部题面（或反过来），两边的写法不是同一串
    token —— 读数会把「题面不一致」记成模型效果。
    """
    tj, ej = getattr(args, 'train_json', None), getattr(args, 'eval_json', None)
    if bool(tj) != bool(ej):
        raise SystemExit(f'{who}：--train-json 与 --eval-json 必须成对给（现在 '
                         f'train={tj} eval={ej}）—— 只给一半就是两边题面不同源')
    if tj and os.path.abspath(tj) == os.path.abspath(ej):
        raise SystemExit(f'{who}：--train-json 与 --eval-json 指向同一个文件 {tj} —— '
                         '训练集当评估集，读数无效')


def _dataset(tok, split, args, *, prompt_only=False, n_vis=None, frontend=None,
             sfrontend=None, ph_id=None, organ_map=None, ev_ph_id=None):
    nv = n_vis if n_vis is not None else _n_vis(args)
    need_feats = (nv != 0) or bool(getattr(args, 'fuse', False))
    ds = JevDataset(tok, split, tuple(args.rules.split(',')),
                    keep_tsv=args.keep_tsv, newq_keep=args.newq_keep,
                    newq_dir=DEFAULT_NEWQ_DIR,
                    cache_root=args.cache_root, cache_sha12=args.cache_sha12,
                    feats=args.feats, n_organs=args.n_organs, frontend=frontend,
                    sfrontend=sfrontend, ph_id=ph_id,
                    organ_crop=args.organ_crop, crop_margin=args.crop_margin,
                    organ_map=organ_map,
                    only_prefixes=tuple(args.only_prefixes.split(',')) if args.only_prefixes else None,
                    check_cache=not args.no_check_cache,
                    n_vis=nv, need_feats=need_feats,
                    prompt_only=prompt_only,
                    fusion_pool=int(getattr(args, 'fusion_pool', 0) or 0),
                    organ2d=bool(getattr(args, 'organ2d', False)),
                    organ_evidence=bool(getattr(args, 'organ_evidence', False)),
                    organ_route=bool(getattr(args, 'organ_route', False)),
                    organ_acquire=bool(getattr(args, 'organ_acquire', False)),
                    ev_ph_id=ev_ph_id,
                    prompt_no_count=bool(getattr(args, 'prompt_no_count', False)),
                    option_perm=int(getattr(args, 'option_perm', 0) or 0),
                    vis_tokens=str(getattr(args, 'vis_tokens', 'all') or 'all'),
                    slot_perm=int(getattr(args, 'slot_perm', 0) or 0),
                    json_path=_json_path(args, prompt_only))
    _check_infer_form(args, ds)
    return ds


def _check_infer_form(args, ds):
    """A2 的两道闸：缓存的**推理形态**要与声明的身份一致，几何列要与切片同坐标系。

    第一道（identity ↔ cache）：`radar_ft` 与 `radar_ft_nowin` 用的是**同一份权重**
    —— basename / sha256 / 张量数三件核对对它们给不出任何区分，能区分的只有缓存 meta
    里的 `infer_form`。指错根目录不会炸形状（36+1 行、256 维、L2 归一化全都合法），
    只会让「不跑滑窗」这条臂读着跑过滑窗的向量出数，所以必须在装载期拒。

    第二道（geom_frame ↔ box 消费方）：nowin 的 mask 来自 `load_and_prepare_whole`
    定形的那一卷，而切片缓存读的是 `load_and_prepare`（非零区 crop + pad，逐卷变尺寸）
    那一卷 —— 拿后者的层号去配前者的 box，等于每个器官的证据都来自别人的位置。
    正解是编码时 `cache.py --geom-from-root` 把窗口那份几何列搬进 nowin 缓存
    （`geom_frame='window'`）；没搬就是 live 现算或忘了搬，此时只允许跑不读 box 的臂。
    """
    if ds.feats_mode == 'cache' and ds.meta.get('encoder_identity'):
        want = RE.infer_form_of(args.encoder) if args.encoder in RE.ENCODER_IDENTITIES else None
        if want is not None and ds.infer_form != want:
            raise SystemExit(
                f'--encoder={args.encoder} 的推理形态是 {want}，但缓存 '
                f'{ds.shard_dir} 的 meta 写的是 {ds.infer_form} —— 同一份权重、'
                '两种喂法，特征不是同一个函数（形状合法，语义错位）。'
                'A2 请配 --cache-root 指向 nowin 那份缓存；跑主臂请把 --encoder 改回 radar_ft')
    reads_box = (bool(getattr(args, 'organ_evidence', False))
                 or bool(getattr(args, 'organ2d', False))
                 or str(getattr(args, 'organ_crop', 'off')) != 'off')
    if reads_box and getattr(ds, 'geom_frame', 'window') != 'window':
        bad = [n for n, v in (('--organ-evidence', args.organ_evidence),
                              ('--organ2d', getattr(args, 'organ2d', False)),
                              ('--organ-crop', str(args.organ_crop) != 'off')) if v]
        joined = ' / '.join(bad)
        raise SystemExit(
            f'缓存的几何列（organ_box / organ_zprof）在 {ds.geom_frame} 坐标系里，'
            f'而 {joined} 要按层号取切片区 —— 切片缓存读的是 load_and_prepare '
            "那卷（非零区 crop + pad），两者不同源：box 框到的层不是证据所在的那几层。"
            'A2 的缓存要用 `cache.py run --geom-from-root <滑窗缓存根>` 重刷，'
            '把窗口那份 organ_box/organ_zprof/sizes/sag_voxels 搬过来（geom_frame=window）')


def _check_rows(model, ds):
    """序列里的占位符数必须等于 projector 的槽数（state-only 时 projector 不参与）。"""
    if ds.n_vis and ds.n_vis != model.n_tok:
        raise SystemExit(f'数据有 {ds.n_vis} 个视觉占位符，projector 只有 {model.n_tok} 个槽 —— '
                         f'feats={ds.feats_mode} 那一路的行数与模型不一致')


def _check_slot_ablation(ds, who):
    """A1 的槽序置换 / A3 的单槽与「按器官名查 `has_feat` 行号」那套视图不可叠加。

    `views.py` 的 target/distractor/blind/keep-bound 都是**在 npz 行序上**翻 `has_feat`
    （`load_organ_bindings` 返回 `RE.ORGANS.index`），而 A1 之后「第 20 行是肝」不再成立：
    叠加的结果不是报错，是遮住另一批器官 —— 读数会安静地变成另一个反事实的数。
    这四条消融臂都跑 `--loss combined`，所以这里宁可拒也不接。真要那条组合臂，
    改法是让 views 走 `JevDataset._rows_of`（尚未做）。
    """
    vt = getattr(ds, 'vis_tokens', 'all')
    if getattr(ds, 'slot_perm', 0):
        raise SystemExit(f'{who} 与 --slot-perm={ds.slot_perm} 不合：视图的行号按 npz 序算'
                         '（`RE.ORGANS.index`），置换后遮的就不是绑定器官了。要么关掉这一侧的'
                         ' --slot-perm，要么先把 views.py 改成走 dataset 的逆映射')
    if vt == 'whole':
        raise SystemExit(f'{who} 与 --vis-tokens whole 不合：A3 的序列里只有一个整图 token，'
                         '没有「绑定器官的那几行」可供视图增删')


def _check_racd(args, ds):
    """`--loss racd` 的接线核对：视图靠翻 `has_feat` 造，前提是真的有特征可翻。

    这三条都在开训前拦：`feats=zeros` 时 `has_feat` 本来就全 False，三个视图逐位相同，
    两项比较损失恒为 0 —— 前向不报错，跑完得到的是一条「什么也没训」的曲线。
    `n_vis=0`（state-only / organ-slots off）连器官槽都没有，mask 无处可落。
    """
    _check_slot_ablation(ds, '--loss racd 的视图')
    if ds.feats_mode not in ('cache', 'live'):
        raise SystemExit(f'--loss racd 要 --feats cache|live（收到 {ds.feats_mode}）—— '
                         '视图靠把绑定器官的 has_feat 翻成 False 造（projector 换成可学的 '
                         'missing 向量）；zeros/none 下 has_feat 恒 False，M 个视图逐位相同，'
                         '比较项恒 0，跑完是一条什么也没训的曲线')
    if not ds.n_vis:
        raise SystemExit('--loss racd 要 n_vis>0（--mode full 且 --organ-slots on）—— '
                         '序列里没有器官槽就没地方落 mask')
    if args.racd_views < 2:
        raise SystemExit(f'--racd-views={args.racd_views} < 2 —— 单视图没有可比的两端')
    if args.racd_m_rank <= 0 or args.racd_m_null < 0:
        raise SystemExit(f'm_rank={args.racd_m_rank} 要 >0（=0 时排序项退化成恒等式），'
                         f'm_null={args.racd_m_null} 要 >=0')
    if args.racd_lam_blind > 0 and not args.racd_blind:
        raise SystemExit(f'--racd-lam-blind={args.racd_lam_blind} >0 但没开 --racd-blind ⇒ '
                         '批次里一个 blind 槽都没有，这一项恒 0（前向不报错，跑完是一条'
                         '「什么也没训」的曲线，与 λ_corr=0 的 GRPO 臂同一种废臂）')
    if args.racd_blind and args.racd_views < 3:
        raise SystemExit(f'--racd-blind 要 --racd-views>=3（收到 {args.racd_views}）—— '
                         '盲视图恒占最后一槽，剩下的至少要是 full 与 target 两端')
    if args.racd_blind and not args.racd_mask_whole \
            and int(ds.n_vis) == int(ds.n_organs) + 1:
        raise SystemExit('--racd-blind 要 --racd-mask-whole 一起开：这条臂序列里有第 37 个'
                         f'整图槽（n_vis={int(ds.n_vis)} == n_organs+1），不遮它的话盲视图'
                         '仍带 whole_feat ⇒ c_full − c_blind 量到的是泄漏，不是语言先验')
    if args.racd_m_blind < 0:
        raise SystemExit(f'--racd-m-blind={args.racd_m_blind} < 0 —— 间隔为负等于要求'
                         '盲视图比整卷更自信')
    if args.racd_lam_cal < 0:
        raise SystemExit(f'--racd-lam-cal={args.racd_lam_cal} < 0 —— 翻号变成「信心离答对'
                         '概率越远越好」（与 losses.racd_loss 的同号闸重复，为的是少占一次'
                         '4 卡分配）')
    if args.racd_lam_cal > 0 and args.racd_cal_mode == 'pi' and not args.racd_prior_table:
        raise SystemExit('--racd-cal-mode=pi 要 --racd-prior-table：没有逐题 π_q，v(π_q) 无从'
                         '算起。静默退回 flat 会让 armH 变成「armG 换了个名字」——损失照算、'
                         '曲线照降、eval 指标照出，事后完全查不出来')
    if args.racd_lam_cal <= 0 and args.racd_cal_mode == 'pi':
        raise SystemExit('--racd-cal-mode=pi 但 --racd-lam-cal=0 ⇒ cal 项恒 0：这一项没点火，'
                         '跑出来是 armF 的复跑。要么给 λ，要么把 mode 留 flat')
    if args.racd_cal_pi_hi < args.racd_cal_pi_lo:
        raise SystemExit(f'--racd-cal-pi-hi={args.racd_cal_pi_hi} < '
                         f'--racd-cal-pi-lo={args.racd_cal_pi_lo} —— 带是空集，等于「只在带外'
                         '加权」，方向反了')
    if not (0.0 < args.racd_cal_pi_w <= 1.0):
        raise SystemExit(f'--racd-cal-pi-w={args.racd_cal_pi_w} 要落在 (0,1] —— 设 0 等于把带外'
                         '行从这一项里摘干净：armH 相对 armF 就不止差「加权方式」一项，还差了'
                         'batch 构成，两条臂不再是单因子对照')
    if args.racd_cal_warm < 0:
        raise SystemExit(f'--racd-cal-warm={args.racd_cal_warm} < 0 —— 点火窗口为负，ramp 会'
                         '从第 1 步就往回退')
    if not (0.0 <= args.racd_tau <= 1.0) or args.racd_min_frac < 0 or args.racd_max_repeat < 1:
        raise SystemExit(f'分层参数越界：tau={args.racd_tau}∈[0,1] '
                         f'min_frac={args.racd_min_frac}>=0 max_repeat={args.racd_max_repeat}>=1')


def _check_grpo(args, ds):
    """`--loss grpo` 的接线核对。全部在开训前拦 —— 这些错都不会在前向报错，
    只会让跑完的曲线不代表它声称代表的那个臂。

    多视图那部分前提与 `_check_racd` 完全一样（视图靠翻 `has_feat` 造），但**只在
    `λ_corr>0` 时才需要**：RAC 的 rank 轴（组内不同 rollout 之间）是单视图的，
    纯 rank 臂不必付 M 倍前向。
    """
    if args.grpo_samples < 2:
        raise SystemExit(f'--grpo-samples={args.grpo_samples} < 2 —— 组内归一化至少要两个'
                         ' rollout 才有 σ（RAC 用 n=8）')
    if args.grpo_m_rank < 0 or args.grpo_m_corr < 0:
        raise SystemExit(f'margin 不能为负：m_rank={args.grpo_m_rank} m_corr={args.grpo_m_corr}')
    if args.grpo_alpha < 0:
        raise SystemExit(f'--grpo-alpha={args.grpo_alpha} < 0 —— severity 斜率的符号是'
                         '「越退化要求越不自信」，负值会把方向翻过来')
    if args.grpo_w_task < 0:
        raise SystemExit(f'--grpo-w-task={args.grpo_w_task} < 0 —— r^task 是奖励，'
                         '翻号就变成罚正确')
    if args.grpo_w_task == 0 and args.grpo_lam_rank <= 0 and args.grpo_lam_corr <= 0:
        raise SystemExit('--grpo-w-task=0 且 λ_rank=λ_corr=0 ⇒ r̃≡0 ⇒ 每组 sd=0 ⇒ adv≡0，'
                         'pg 零梯度：700 步烧完只会得到一条「RL 没动」的空读数，'
                         '分不清是机制无效还是臂压根没配起来。要纯整形就把 λ_corr 设非零')
    if args.grpo_beta > 0 and not (args.grpo_ref or args.init_from):
        raise SystemExit('--grpo-beta>0 要一个参考策略：给 --grpo-ref <SFT ckpt>。'
                         '续跑时 --init-from 必须丢掉（它会把 RL 段的进度覆盖回 SFT 起点），'
                         '锚就得自己带着 —— 否则 deepcopy 到的是**已经漂过**的策略，'
                         'KL 变成「别离开 500 步前的自己」，那不是锚')
    if not 0.0 <= args.grpo_kind_alpha <= 1.0:
        raise SystemExit(f'--grpo-kind-alpha={args.grpo_kind_alpha} 应在 [0,1]：'
                         '1=按行数（现状），0=三个题型等分曝光')
    if args.grpo_kind_alpha < 1.0 and args.grpo_balance != 'group':
        raise SystemExit(f'--grpo-kind-alpha={args.grpo_kind_alpha} < 1 要 --grpo-balance group '
                         '（题型平权叠在绑定器官分层之上，两者不是二选一）')
    if args.grpo_lam_corr > 0 or args.grpo_balance == 'group':
        _check_slot_ablation(ds, 'GRPO 的多视图/分层采样')
        if ds.feats_mode not in ('cache', 'live'):
            raise SystemExit(f'--grpo-lam-corr>0 / --grpo-balance group 要 --feats cache|live'
                             f'（收到 {ds.feats_mode}）—— 视图靠把绑定器官的 has_feat 翻成 '
                             'False 造；zeros/none 下 has_feat 恒 False，M 个视图逐位相同，'
                             'r^corr 恒 0，白付 M 倍前向')
        if not ds.n_vis:
            raise SystemExit('--grpo-lam-corr>0 / --grpo-balance group 要 n_vis>0'
                             '（--mode full 且 --organ-slots on）—— 序列里没有器官槽就'
                             '没地方落 mask')
        if args.racd_views < 2:
            raise SystemExit(f'--racd-views={args.racd_views} < 2 —— r^corr 要干净/退化两端')
        if not (0.0 <= args.racd_tau <= 1.0) or args.racd_min_frac < 0 \
                or args.racd_max_repeat < 1:
            raise SystemExit(f'分层参数越界：tau={args.racd_tau}∈[0,1] '
                             f'min_frac={args.racd_min_frac}>=0 '
                             f'max_repeat={args.racd_max_repeat}>=1')
    if args.grpo_view_weight == 'grounded':
        if not (args.grpo_lam_corr > 0 or args.grpo_balance == 'group'):
            raise SystemExit('--grpo-view-weight grounded 要多视图批次：加 --grpo-balance group '
                             '（或 --grpo-lam-corr>0）—— 单视图臂不套 MultiViewDataset，'
                             '批次里压根没有 blind 槽')
        if not args.racd_blind:
            raise SystemExit('--grpo-view-weight grounded 要 --racd-blind：它拿 blind 槽当'
                             '先验基线来量其它视图的 grounding，没有 blind 槽这一项无定义')
        if args.racd_views < 3:
            raise SystemExit(f'--grpo-view-weight grounded 要 --racd-views>=3（收到 '
                             f'{args.racd_views}）—— blind 恒占最后一槽，剩下至少 full+target')
        if not args.racd_mask_whole and int(ds.n_vis) == int(ds.n_organs) + 1:
            raise SystemExit('--grpo-view-weight grounded 要 --racd-mask-whole：这条臂有第 37 '
                             f'个整图槽（n_vis={int(ds.n_vis)}==n_organs+1），blind 不遮它'
                             '量到的是泄漏不是语言先验（同 racd blind 闸）')


def _check_dpo(args, ds):
    """`--loss dpo` 的接线核对（配置期拦，不白占卡）。跨视图硬负例 DPO 要 full+blind
    两端：blind 提供先验答案（rejected 标签），full 提供 chosen/rejected 的 logπ。"""
    _check_slot_ablation(ds, '--loss dpo 的视图')
    if ds.feats_mode not in ('cache', 'live'):
        raise SystemExit(f'--loss dpo 要 --feats cache|live（收到 {ds.feats_mode}）—— '
                         'blind 视图靠把在场槽的 has_feat 翻 False 造；zeros/none 下盲视图'
                         '与 full 逐位相同，先验答案=证据答案，偏好对恒空，跑完什么也没训')
    if not ds.n_vis:
        raise SystemExit('--loss dpo 要 n_vis>0 —— 序列里没有器官槽就没地方落 blind mask')
    if not args.racd_blind:
        raise SystemExit('--loss dpo 要 --racd-blind：rejected 答案取自 blind 视图的 argmax，'
                         '没有 blind 槽就没有先验答案可配')
    if args.racd_views < 3:
        raise SystemExit(f'--loss dpo 要 --racd-views>=3（收到 {args.racd_views}）—— '
                         'blind 恒占最后一槽，plan_views 至少给 full+target+blind 三槽；'
                         'dpo 读 full 与 blind，target 槽不进损失')
    if not args.racd_mask_whole and int(ds.n_vis) == int(ds.n_organs) + 1:
        raise SystemExit('--loss dpo 要 --racd-mask-whole：这条臂有第 37 个整图槽'
                         f'（n_vis={int(ds.n_vis)}==n_organs+1），blind 不遮它量到的是泄漏'
                         '不是语言先验（同 racd blind 闸）')
    if args.dpo_beta <= 0:
        raise SystemExit(f'--dpo-beta={args.dpo_beta} <= 0 —— DPO 要参考锚（beta>0）')
    if not (args.dpo_ref or args.init_from):
        raise SystemExit('--loss dpo 要参考策略：给 --dpo-ref <SFT ckpt>（续跑）'
                         '或 --init-from（首跑，deepcopy 即锚）')
    if args.dpo_lam_ce < 0:
        raise SystemExit(f'--dpo-lam-ce={args.dpo_lam_ce} < 0 —— 翻号变成罚正确')
    if not (0.0 <= args.racd_tau <= 1.0) or args.racd_min_frac < 0 or args.racd_max_repeat < 1:
        raise SystemExit(f'分层参数越界：tau={args.racd_tau}∈[0,1] '
                         f'min_frac={args.racd_min_frac}>=0 max_repeat={args.racd_max_repeat}>=1')


def _check_evgrpo(args, ds):
    """`--loss evgrpo`（armEV 证据获取 RL）的接线核对，全部开训前拦。

    与 grpo 的范式差：动作是**证据子集**不是答案候选，所以前提落在证据臂 + 获取头 +
    在场性掩码杠杆上，而不是多视图反事实。这些错都不在前向报错，只会让跑完的曲线
    不代表它声称代表的那个臂。
    """
    if args.ev_samples < 2:
        raise SystemExit(f'--ev-samples={args.ev_samples} < 2 —— 组内归一化至少要两个'
                         '证据子集才有 σ')
    if not getattr(args, 'organ_acquire', False):
        raise SystemExit('--loss evgrpo 要 --organ-acquire（获取策略头）—— 没有头就没有'
                         '可训的动作策略，pg 零梯度')
    if not getattr(args, 'organ_evidence', False):
        raise SystemExit('--loss evgrpo 要 --organ-evidence：获取动作掩的是证据槽')
    _check_slot_ablation(ds, 'armEV 的获取掩码')
    if ds.feats_mode not in ('cache', 'live'):
        raise SystemExit(f'--loss evgrpo 要 --feats cache|live（收到 {ds.feats_mode}）—— '
                         '获取掩码靠把未取槽的 has_feat 翻 False 造；zeros/none 下 '
                         'has_feat 恒 False，G 个子集逐位相同，adv≡0，白付 G 倍前向')
    if not ds.n_vis:
        raise SystemExit('--loss evgrpo 要 n_vis>0（--mode full 且 --organ-slots on）')
    if int(ds.n_vis) != int(ds.n_organs) + 1:
        raise SystemExit(f'--loss evgrpo 要整图槽（n_vis={int(ds.n_vis)} 须 == '
                         f'n_organs+1={int(ds.n_organs) + 1}）—— 整图是获取动作的一个'
                         '候选（acquirable，代价 1 单位），缺它动作空间就不完整')
    if args.ev_lam_cost < 0 or args.ev_beta_cal < 0 or args.ev_lam_task < 0:
        raise SystemExit(f'奖励/监督权重不能为负：lam_cost={args.ev_lam_cost} '
                         f'beta_cal={args.ev_beta_cal} lam_task={args.ev_lam_task}')
    if args.ev_beta_kl < 0:
        raise SystemExit(f'--ev-beta-kl={args.ev_beta_kl} < 0 —— KL 锚强度翻号变成鼓励偏离')
    if args.ev_beta_kl > 0 and not (args.ev_ref or args.init_from):
        raise SystemExit('--ev-beta-kl>0 要一个参考策略：给 --ev-ref <SFT ckpt>。'
                         '续跑时 --init-from 必须丢掉（它会把 RL 段进度覆盖回 SFT 起点），'
                         '锚就得自己带着 —— 否则 deepcopy 到的是已经漂过的策略')
    if args.ev_lam_cost == 0 and args.ev_beta_cal == 0 and args.ev_lam_task == 0:
        raise SystemExit('lam_cost=beta_cal=lam_task=0 ⇒ r≡correctness 且无监督 ⇒ 获取头'
                         '只靠正确性整形、答案头无锚 —— 至少要留一项非零')
    if not (0.0 <= args.racd_tau <= 1.0) or args.racd_min_frac < 0 \
            or args.racd_max_repeat < 1:
        raise SystemExit(f'分层参数越界：tau={args.racd_tau}∈[0,1] '
                         f'min_frac={args.racd_min_frac}>=0 max_repeat={args.racd_max_repeat}>=1')


def _grpo_info(args, samp_info, bind_info, n_views, mask_whole=False):
    """run_info.json 的 `grpo` 块：这条臂的每一个 RL 旋钮都记下来（含哪些取自 RAC、
    哪些是本仓库自己定的），事后对不上账时能一眼看出是哪一项不同。

    `n_views` / `mask_whole` 由调用点给**实际生效**的值：纯 rank 单视图臂（λ_corr=0 且
    balance=off）不套 MultiViewDataset，这时照抄 --racd-views / --racd-mask-whole 会在
    run_info 里留一个从没跑过的口径。
    """
    from .losses import GRPO_ADV_EPS
    return {
        'stage': 2, 'paper': 'RAC arXiv:2605.16999 Eq.1/4-6/7-9/11',
        'n_samples': int(args.grpo_samples), 'beta_kl': float(args.grpo_beta),
        'lam_task': float(args.grpo_lam_task),
        'w_task': float(args.grpo_w_task),
        'lam_rank': float(args.grpo_lam_rank), 'm_rank': float(args.grpo_m_rank),
        'lam_corr': float(args.grpo_lam_corr), 'm_corr': float(args.grpo_m_corr),
        'alpha_sev': float(args.grpo_alpha), 'gate_corr': not args.grpo_no_gate_corr,
        'corr_cand': str(args.grpo_corr_cand),
        'task_views': ('all' if args.grpo_task_views is None
                       else str(args.grpo_task_views)),
        'view_weight': str(args.grpo_view_weight),
        'balance': str(args.grpo_balance), 'n_views': int(n_views),
        'kind_alpha': float(args.grpo_kind_alpha),
        'mask_whole': bool(mask_whole),
        'from_paper': {'n_samples': 8, 'm_rank': 0.05, 'lam_rank': 0.2, 'lam_corr': 0.3},
        'ours': {'m_corr': float(args.grpo_m_corr), 'alpha_sev': float(args.grpo_alpha),
                 'beta_kl': float(args.grpo_beta), 'adv_eps': float(GRPO_ADV_EPS),
                 'kind_alpha': float(args.grpo_kind_alpha)},
        'ref': (os.path.abspath(args.grpo_ref or args.init_from)
                if (args.grpo_beta > 0 and (args.grpo_ref or args.init_from)) else None),
        'sampler': samp_info, 'bindings': bind_info,
    }


def _accum_diag(diag, num, den, weights):
    """把一个 micro-batch 的 diag 累进窗口加权均值（racd 与 grpo 共用）。

    `weights` 见 losses.RACD_DIAG_W / GRPO_DIAG_W：值 None = 等权，值是个键名 = 按
    `diag[那个键]` 加权。存在的理由是那些**只在部分样本上才有定义**的键 —— racd 的
    Δ_target 只在抽得出反事实的组上有定义，grpo 的 viol_corr/gate_corr 只在配得出
    干净/退化对的槽上有定义。等权会让 bs=1 时大量「这一批没有」的 0 混进均值，把读数
    系统性压向 0 —— 那正好是「机制没生效」和「机制没被触发」分不开的那个方向。
    """
    for k, v in diag.items():
        wk = weights.get(k)
        w = 1.0 if wk is None else float(diag[wk])
        num[k] += float(v) * w
        den[k] += w


def _kind3(rd, suf, fmt='{:.2f}'):
    """`noul/choice/score` 三个读数的紧凑三连。缺键打 NA（该题型本窗口没出现）。"""
    from .losses import GRPO_KINDS
    vals = [rd.get(f'{k}_{suf}') for k in GRPO_KINDS]
    return '/'.join('NA' if v is None else fmt.format(v) for v in vals)


def _grpo_extra(rd):
    """train_log 一行的 grpo 摘要。`rd` 是窗口均值，**键可能缺**（见 `_accum_diag`）。

    `adv_nz` 是这条臂的命门：σ=0 的组零梯度，全对/全错的题白跑一趟前向。它掉下来就是
    「课程饱和了」，不是「学不动了」。`gate_corr` 缺键 = 本窗口一组干净/退化对都没配上，
    这时打 0.000 会被读成「质量闸全程关着」，必须打 NA。

    批均值 `adv_nz`/`hit` 必须配着后面的三连读：三种题型的**随机基线不同**（noul 0.5、
    score 0.2、choice 1/K），混算的 `hit` 没有地板可对；`adv_nz` 的饱和机制也不同
    （choice 的 accept_mask 让「奖励相同」的集合变大 ⇒ 结构性更容易 σ=0）。批均值正常
    而某一个题型是 0.00，说明那一支根本没在学 —— 只看批均值是看不出来的。
    """
    g = rd.get('gate_corr')
    vw = rd.get('vw_ground_frac')
    return (f' pg={rd.get("pg", 0.0):+.4f} r̃={rd.get("r_mean", 0.0):+.4f} '
            f'adv_nz={rd.get("adv_nz", 0.0):.3f} hit={rd.get("hit", 0.0):.3f} '
            + (f'gate={g:.3f}' if g is not None else 'gate=NA')
            + (f' vw_ground={vw:.3f} vw_full={rd.get("vw_full", 0.0):.3f}'
               if vw is not None else '')
            + f' | nz={_kind3(rd, "adv_nz")} hit={_kind3(rd, "hit")}'
            f' sh={_kind3(rd, "share")}')


def _dpo_extra(rd):
    """train_log 一行的 dpo 摘要。`margin` 是隐式奖励差（chosen−rejected 的参考 log-ratio），
    应当随训练上升；`gate_frac` 是「先验≠gold」的配对率（机制触发率），它≈0 说明先验本来
    就答对、没有冲突可修。`dpo`/`margin` 缺键 = 本窗口一个偏好对都没配上（按 n_pairs 加权）。"""
    d = rd.get('dpo')
    mg = rd.get('margin')
    return (f' dpo={d:.4f}' if d is not None else ' dpo=NA') \
        + (f' margin={mg:+.4f}' if mg is not None else ' margin=NA') \
        + (f' gate={rd.get("gate_frac", 0.0):.3f} '
           f'q_full={rd.get("q_full", 0.0):.3f} q_blind={rd.get("q_blind", 0.0):.3f}')


def _evgrpo_info(args, samp_info, bind_info):
    """run_info.json 的 `evgrpo` 块：armEV 的每个 RL 旋钮都记下来（获取动作空间、奖励
    三项权重、答案监督与 KL 锚），事后对不上账时能一眼看出是哪一项不同。

    与 grpo 块的范式差：动作是证据子集（输入侧），不是答案候选；奖励 = correctness −
    λ_cost·|S| − β_cal·Brier；答案头由 λ_task·combined_loss 微调、β_kl 弱锚。
    """
    from .losses import GRPO_ADV_EPS
    return {
        'stage': 2, 'arm': 'armEV evidence-acquisition GRPO + Brier calibration',
        'action': 'bernoulli organ-evidence subset (present organs + whole slot)',
        'n_samples': int(args.ev_samples),
        'lam_cost': float(args.ev_lam_cost), 'beta_cal': float(args.ev_beta_cal),
        'beta_kl': float(args.ev_beta_kl), 'lam_task': float(args.ev_lam_task),
        'balance': str(args.grpo_balance),
        'reward': 'correctness - lam_cost*|S| - beta_cal*Brier',
        'adv_eps': float(GRPO_ADV_EPS),
        'from_paper': {'n_samples': 8},
        'ours': {'lam_cost': float(args.ev_lam_cost), 'beta_cal': float(args.ev_beta_cal),
                 'beta_kl': float(args.ev_beta_kl), 'lam_task': float(args.ev_lam_task)},
        'ref': (os.path.abspath(args.ev_ref or args.init_from)
                if (args.ev_beta_kl > 0 and (args.ev_ref or args.init_from)) else None),
        'sampler': samp_info, 'bindings': bind_info,
    }


def _evgrpo_extra(rd):
    """train_log 一行的 evgrpo 摘要。`rd` 是窗口均值，键可能缺（见 `_accum_diag`）。

    命门读数：`adv_nz`（σ=0 的组零梯度，掉下来=课程饱和）；`acq_cost`（→0=语言先验塌方、
    →n_present=代价太弱）；`acq_bound_frac`（获取的器官里绑定占比，应>随机）；
    `acq_whole_frac`（≈1=整图捷径塌方）；`acq_brier`/`pi_max`（校准是否随训练移动）。
    批均值配题型三连（三种题型随机基线不同，混算 hit 没有地板可对）。
    """
    return (f' pg={rd.get("pg", 0.0):+.4f} r̃={rd.get("r_mean", 0.0):+.4f} '
            f'adv_nz={rd.get("adv_nz", 0.0):.3f} hit={rd.get("hit", 0.0):.3f} '
            f'cost={rd.get("acq_cost", 0.0):.2f} brier={rd.get("acq_brier", 0.0):.3f} '
            f'bound={rd.get("acq_bound_frac", 0.0):.3f} '
            f'whole={rd.get("acq_whole_frac", 0.0):.3f} '
            f'pi_max={rd.get("pi_max", 0.0):.3f}'
            + f' | nz={_kind3(rd, "adv_nz")} hit={_kind3(rd, "hit")}'
            f' sh={_kind3(rd, "share")}')


def _native_channel(args, tok, model):
    """B 支的接线核对：宿主有塔、插槽 id 在词表内、器官槽换一个不撞的 id、塔按声明冻结。

    返回 `(sfrontend, ph_id, ev_ph_id)`；`--slices 0` 时 `(None, None, None)` = A 支一条
    通道。`ev_ph_id` 只在 --organ-evidence 下非 None（第二个空闲占位 id）。
    这些都在序列成形之前查 —— 撞 id 的后果是两次 `masked_scatter` 互相覆盖，
    前向不报错；id 越界的后果要等到第一个 batch 才炸。
    """
    if getattr(args, 'pure2d', False):
        if args.mode != 'full':
            raise SystemExit('--pure2d 只能是 --mode full（state-only 连切片都不要，'
                             '那是 ① 级地板，跑 floors）')
        if not args.slices:
            raise SystemExit('--pure2d 要 --slices>0 —— 纯 2D 的唯一视觉入口是宿主原生切片')
        if args.feats != 'none':
            raise SystemExit(f'--pure2d 要 --feats none（收到 {args.feats}）—— 3D 特征'
                             '一条都不读，缓存/前端都不该被打开')
        if args.fuse:
            raise SystemExit('--pure2d 与 --fuse 不合：融合的 K/V 来自 3D 特征')
        if str(getattr(args, 'organ_slots', 'on')) != 'off':
            raise SystemExit('--pure2d 要 --organ-slots off —— 纯 2D 序列里没有器官槽')
        if args.organ_crop != 'off':
            raise SystemExit('--pure2d 与 --organ-crop 不合：裁剪臂读的是 3D 定位列')
        if args.freeze_llm:
            raise SystemExit('--pure2d 与 --freeze-llm 不合：projector 已冻、器官槽已关，'
                             '冻结 LLM 后没有任何可训参数 —— 直接全参微调请去掉 --freeze-llm')
    fp = int(getattr(args, 'fusion_pool', 0) or 0)
    if fp:
        if args.mode == 'state-only' or str(getattr(args, 'organ_slots', 'on')) == 'off':
            raise SystemExit('--fusion-pool 的 query 来自器官槽 —— state-only / '
                             '--organ-slots off 时没有 query，池化无从谈起')
        if not args.slices:
            raise SystemExit('--fusion-pool 要 --slices>0：池化的输入是切片 token')
        if model.pool is None or int(fp) != model.pool.n_tok:
            raise SystemExit(f'--fusion-pool={fp} 与模型池化数不一致 —— 每个器官一个摘要，'
                             f'N 必须等于器官数（{model.n_organs}）')
    if getattr(args, 'organ2d', False):
        if args.mode == 'state-only':
            raise SystemExit('--organ2d 配 --mode state-only 无意义：器官槽不在序列里，'
                             '残差无处注入 —— 那是 ① 级地板')
        if str(getattr(args, 'organ_slots', 'on')) == 'off':
            raise SystemExit('--organ2d 配 --organ-slots off：器官槽不在序列里，残差无处注入')
        if not args.slices:
            raise SystemExit('--organ2d 要 --slices>0 —— 2D 证据来自宿主塔对切片的前向')
        if args.fuse:
            raise SystemExit('--organ2d 与 --fuse 不合：organ2d 的序列里没有切片 token，'
                             '融合没有 Q 流；2D 证据走残差注入，不占序列')
        if args.feats == 'live' and args.no_write_through:
            raise SystemExit('--organ2d 配 --feats live --no-write-through：定位列 '
                             '（organ_box）不落盘、无处可读 —— 要么开 write-through，'
                             '要么走 cache')
        if model.organ2d is None:
            raise SystemExit('--organ2d 给了但模型没建 Organ2DResidual —— _build_model 与 '
                             '_native_channel 的接线不一致')
    if getattr(args, 'organ_evidence', False):
        if args.mode == 'state-only':
            raise SystemExit('--organ-evidence 配 --mode state-only 无意义：器官槽不在'
                             '序列里，证据 token 无槽可依 —— 那是 ① 级地板')
        if str(getattr(args, 'organ_slots', 'on')) == 'off':
            raise SystemExit('--organ-evidence 配 --organ-slots off：器官槽不在序列里，'
                             '证据 token 无槽可依')
        if not args.slices:
            raise SystemExit('--organ-evidence 要 --slices>0 —— 证据的 K/V 来自宿主塔'
                             '对切片的前向')
        if args.fuse:
            raise SystemExit('--organ-evidence 与 --fuse 不合：两条臂都改器官段与切片段'
                             '的耦合方式，叠加的语义没人定义过 —— 各自立项')
        if args.feats == 'zeros':
            raise SystemExit('--organ-evidence 配 --feats zeros 只验流水线，不许报数'
                             '（zeros 没有 organ_box，区域归属恒空）')
        if args.feats == 'live' and args.no_write_through:
            raise SystemExit('--organ-evidence 配 --feats live --no-write-through：定位列 '
                             '（organ_box）不落盘、无处可读 —— 要么开 write-through，'
                             '要么走 cache')
        if model.evidence is None:
            raise SystemExit('--organ-evidence 给了但模型没建 OrganEvidence —— '
                             '_build_model 与 _native_channel 的接线不一致')
    if getattr(args, 'organ_route_visual', False) or getattr(
            args, 'organ_route_perorgan', False):
        args.organ_route = True
    if getattr(args, 'organ_route_visual', False) and getattr(
            args, 'organ_route_perorgan', False):
        raise SystemExit('--organ-route-visual 与 --organ-route-perorgan 互斥 —— '
                         '同一个门控只能有一种权重参数化')
    if getattr(args, 'organ_route', False):
        if args.mode == 'state-only':
            raise SystemExit('--organ-route 配 --mode state-only 无意义：n_vis=0，'
                             '序列里没有器官槽可门控 —— 那是 ① 级地板')
        if str(getattr(args, 'organ_slots', 'on')) == 'off':
            raise SystemExit('--organ-route 配 --organ-slots off：器官槽不在序列里，'
                             '门控无槽可缩放')
        if model.route is None:
            raise SystemExit('--organ-route 给了但模型没建 OrganRoute —— '
                             '_build_model 的接线不一致')
        want = ('visual' if getattr(args, 'organ_route_visual', False)
                else 'perorgan' if getattr(args, 'organ_route_perorgan', False)
                else 'scalar')
        got = getattr(model.route, 'kind', None)
        if got != want:
            raise SystemExit(f'--organ-route 期望 {want} 门控，模型建的是 {got} —— '
                             '_build_model 的 route 接线不一致')
    if args.organ_crop != 'off':
        if args.mode == 'state-only':
            raise SystemExit('--organ-crop 配 --mode state-only 无意义：n_vis=0 时两条'
                             '视觉通道都关着，无物可裁（state-only 是 ① 级地板）')
        if not args.slices:
            raise SystemExit('--organ-crop 要 --slices>0 —— 切片流不存在时裁剪无物可裁')
        if args.feats == 'zeros':
            raise SystemExit('--organ-crop 配 --feats zeros 只验流水线，不许拿来报数'
                             '（与 A 支同口径）—— 要冒烟用真缓存，或 live 的小规模 dry')
        if args.feats == 'live' and args.no_write_through:
            raise SystemExit('--organ-crop 配 --feats live --no-write-through：定位列'
                             '（organ_box / organ_zprof）不落盘、无处可读 —— 要么开 '
                             'write-through，要么走 cache')
    if args.organ_slots == 'off':
        if args.mode == 'state-only':
            raise SystemExit('--organ-slots off 配 --mode state-only 无意义：那档 n_vis=0 '
                             '且融合也被拒，两条视觉通道全关 —— 那是 ① 级地板，跑 floors')
        if not args.fuse and not getattr(args, 'pure2d', False):
            raise SystemExit('--organ-slots off 必须配 --fuse：器官槽关掉后 3D 的唯一入口'
                             '就是融合的 K/V，两个都不开 = 纯 2D 宿主基线（另一条臂，'
                             '请走 --pure2d）—— 对齐段（--freeze-llm）也会因为 projector '
                             '不进计算图而空转')
    sf = _build_sfrontend(args)
    if args.fuse and args.mode == 'state-only':
        raise SystemExit('--fuse 配 --mode state-only 无意义：n_vis=0 时两条视觉通道都关着，'
                         'state-only 那一档是 ① 级地板，融合没有序列可注')
    if sf is None:
        if args.freeze_vision:
            raise SystemExit('--freeze-vision 要配 --slices：原生通道没开时它无物可冻 '
                             '（RADAR 那一支天生冻结，不需要这个开关）')
        if args.fuse:
            raise SystemExit('--fuse 要配 --slices>0 —— 融合的 Q 流（切片 token）来自原生'
                             '通道，没有它融合无物可注')
        vis = getattr(getattr(model.llm, 'model', model.llm), 'visual', None)
        if vis is not None:
            for p in vis.parameters():
                p.requires_grad_(False)
            n_vis = sum(p.numel() for p in vis.parameters())
            print(f'[slice] 原生通道关闭：宿主视觉塔 {n_vis / 1e6:.1f}M 参数冻结'
                  '（不进前向/优化器/ckpt）', flush=True)
        return None, None, None
    if not model.has_vision:
        raise SystemExit(f'--slices={args.slices} 要宿主自带视觉塔，但 --llm={args.llm} 是'
                         '纯文本基座（config 没有 vision_config）—— B 支请指 pretrained/Qwen3.5-*')
    vocab = model.llm.get_input_embeddings().weight.shape[0]
    if max(sf.img_ids) >= vocab:
        raise SystemExit(f'宿主的插槽 id {sf.img_ids} 有一个 >= 嵌入词表 {vocab} —— '
                         '--tok 与 --llm 必须同源（同一套 Qwen3.5 词表），否则序列里的'
                         '占位 id 根本不存在')
    ph_id = model.radar_slot_id(len(tok))
    if ph_id in sf.img_ids:
        raise SystemExit(f'RADAR 槽的占位 id {ph_id} 与宿主插槽撞了 —— 换一个空闲 id')
    ev_ph_id = None
    if getattr(args, 'organ_evidence', False):
        ev_ph_id = int(ph_id) + 1
        if ev_ph_id >= vocab:
            raise SystemExit(f'证据槽 id {ev_ph_id} >= 嵌入词表 {vocab} —— 空闲段只够'
                             '一个槽，换一个 id 或换宿主')
        if ev_ph_id in sf.img_ids or ev_ph_id == model.image_token_id:
            raise SystemExit(f'证据槽 id {ev_ph_id} 与宿主插槽撞了 —— 换一个空闲 id')
        if ev_ph_id == ph_id:
            raise SystemExit(f'证据槽 id {ev_ph_id} == 器官槽 id —— 两种插槽会互相覆盖')
    if args.freeze_vision:
        vis = getattr(getattr(model.llm, 'model', model.llm), 'visual', None)
        if vis is None:
            raise SystemExit('--freeze-vision 但宿主里没有 visual 模块 —— 原生通道无从冻结')
        for p in vis.parameters():
            p.requires_grad_(False)
        n_vis = sum(p.numel() for p in vis.parameters())
        print(f'[slice] 宿主视觉塔冻结（{n_vis / 1e6:.1f}M 参数不动）', flush=True)
    if getattr(args, 'pure2d', False):
        slots_note = '纯 2D：无器官槽、无融合、无 3D 特征'
    else:
        slots_note = '器官槽进序列' if args.organ_slots == 'on' else '器官槽关（3D 只作融合 K/V）'
    print(f'[slice] 原生通道 {args.slices} 片/例：processor={sf.info["processor"]} '
          f'patch={sf.info["patch_size"]} merge={sf.merge} 插槽={sf.img_ids}，'
          f'器官槽 id={ph_id}，证据槽 id={ev_ph_id}，{slots_note}，塔冻结={args.freeze_vision}，'
          f'融合={args.fuse}，裁剪={args.organ_crop}', flush=True)
    return sf, ph_id, ev_ph_id


def _run_info(args, tok_path, model, llm_ident, ds_desc, extra=None, frontend=None,
              sfrontend=None, ph_id=None, organ_map_info=None, ev_ph_id=None):
    info = {
        'when': time.strftime('%F %T'),
        'argv': sys_argv(),
        'rules': args.rules.split(','),
        'mode': args.mode, 'feats': args.feats,
        'train_split': args.train_split,
        'cache_root': args.cache_root, 'cache_sha12': args.cache_sha12,
        'limit': args.limit, 'seed': args.seed,
        'tokenizer': tok_path,
        'llm': llm_ident,
        'model': model.describe(),
        'data': ds_desc,
        'notes': [],
    }
    if frontend is not None:
        info['visual_frontend'] = frontend.describe()
    elif args.feats == 'live':
        info['visual_frontend'] = {'visual_mode': 'live', 'encoder_identity': args.encoder,
                                   'built': False,
                                   'note': 'mode=state-only：n_vis=0，前端未载入'}
    info['native_slices'] = int(sfrontend.n_slices) if sfrontend is not None else 0
    info['fuse'] = bool(getattr(args, 'fuse', False))
    info['fusion_pool'] = int(getattr(args, 'fusion_pool', 0) or 0)
    info['organ2d'] = bool(getattr(args, 'organ2d', False))
    info['organ_evidence'] = bool(getattr(args, 'organ_evidence', False))
    info['evidence_no_box'] = bool(info['organ_evidence']
                                   and getattr(args, 'evidence_no_box', False))
    info['organ_route'] = bool(getattr(args, 'organ_route', False))
    info['organ_route_visual'] = bool(getattr(args, 'organ_route_visual', False))
    info['organ_route_perorgan'] = bool(getattr(args, 'organ_route_perorgan', False))
    if info['organ_route_visual'] and not info['organ_route']:
        raise SystemExit('--organ-route-visual 必须同时开 --organ-route'
                         '（CLI 别名正常会自动补；这里报说明内部调用漏了）')
    if info['organ_route_perorgan'] and not info['organ_route']:
        raise SystemExit('--organ-route-perorgan 必须同时开 --organ-route'
                         '（CLI 别名正常会自动补；这里报说明内部调用漏了）')
    if info['organ_route_visual'] and info['organ_route_perorgan']:
        raise SystemExit('--organ-route-visual 与 --organ-route-perorgan 互斥 —— '
                         '同一个门控只能有一种权重参数化')
    info['organ_acquire'] = bool(getattr(args, 'organ_acquire', False))
    if info['organ_acquire'] and not info['organ_evidence']:
        raise SystemExit('--organ-acquire 必须同时开 --organ-evidence（获取动作掩的是证据槽）')
    info['prompt_no_count'] = bool(getattr(args, 'prompt_no_count', False))
    if info['prompt_no_count']:
        info['notes'].append(
            '题面口径 = 无数量词：切片句一律是「You are given axial CT slices」'
            '（导出文件仍含 32，装载时改写；训练/评估同路）。与已登记旧臂'
            '（含「32」逐字段面）相差题面一个 token —— 跨口径不许直接对读数，'
            '对照臂须同口径重跑或同口径重评。')
    info['option_perm'] = int(getattr(args, 'option_perm', 0) or 0)
    if info['option_perm']:
        info['notes'].append(
            f'选项轮转 = seed {info["option_perm"]}：choice 题的选项描述按 '
            'sha256(seed, qid) 在字母位上轮转（字母/尾行/K 不变）。noul 与 score 不动。'
            '与 seed 0 的臂相差 choice 题面 ⇒ 跨口径不许直接对读数。')
    info['organ_slots'] = str(getattr(args, 'organ_slots', 'on'))
    info['vis_tokens'] = str(getattr(args, 'vis_tokens', 'all') or 'all')
    info['organ_index'] = str(getattr(args, 'organ_index', 'on'))
    info['slot_perm'] = int(getattr(args, 'slot_perm', 0) or 0)
    if info['vis_tokens'] != 'all':
        pn, phw, nv = _proj_geom(args, None)
        info['notes'].append(
            f'视觉 token 子集 = --vis-tokens {info["vis_tokens"]}：序列里 {nv} 个 3D token'
            f'（projector 建成 n_organs={pn}, has_whole={phw}）。'
            + ('A4：整图 token 已去掉，organ_evidence 的整图回退支同时消失（无绑定题退回'
               '器官向量本身，计 oev_no_whole_slot）—— 与主臂的差不止「少一行」，'
               '还差那一支回退。' if info['vis_tokens'] == 'organs' else
               'A3：只剩那一个整图 token —— 切片通道、证据 token、融合、路由全关，'
               '与主臂比是「1 个 vs 37 个 + 证据」的整体替换，不是单变量。')
            + '对照臂 = 同参数 --vis-tokens all。')
    if info['organ_index'] == 'off' or info['slot_perm']:
        info['notes'].append(
            f'A1（去解剖索引）= organ_index={info["organ_index"]} / slot_perm='
            f'{info["slot_perm"]}：'
            + ('身份嵌入 = 单个共享向量（可被 MLP 末层 bias 吸收 ⇒ 没有任何一列携带器官身份）'
               if info['organ_index'] == 'off' else '身份嵌入仍按槽位存在（未撤）')
            + ('；槽序按 study 由 sha256 置换（organ_box/zprof 跟随，按名查行走逆映射）'
               if info['slot_perm'] else '；槽序未打乱（固定行序 = 索引仍在）')
            + '。36 行向量、MLP、missing 占位一律未动 —— 撤的是索引，不是内容。'
              '两半都在才是这条臂；对照臂 = 同参数 --organ-index on --slot-perm 0。')
    info['pure2d'] = bool(getattr(args, 'pure2d', False))
    info['train_json'] = getattr(args, 'train_json', None)
    info['eval_json'] = getattr(args, 'eval_json', None)
    info['loss'] = str(getattr(args, 'loss', 'ce'))
    if info['train_json']:
        info['data_source'] = 'export_json'
        train_obj = ('训练对单 token 候选集受限打分（combined：choice CE+可接受集质量+'
                     'Brier / noul BCE / score NLL+RPS）'
                     if info['loss'] == 'combined' else
                     '训练只对 assistant 应答段算 answer-only token CE（Vision-Jev 形式）')
        info['notes'].append(
            f'数据面=导出 JSON：train={info["train_json"]} eval={info["eval_json"]}。'
            f'user 内容逐字取导出文件的题面，{train_obj}；'
            '评估仍走 score_exact() 的逐候选 NLL，两边前缀同源。')
    else:
        info['data_source'] = 'keep_tsv'
    info['grad_ckpt'] = bool(getattr(args, 'grad_ckpt', False))
    if info['grad_ckpt']:
        info['notes'].append(
            'grad_ckpt=on：LLM 逐层梯度检查点（use_reentrant=False）。jjt_mllm 缺 '
            'fla/causal-conv1d，Qwen3.5 线性注意力的 torch 兜底实现反向中间量过大，'
            '不开会在 bs=1 下 OOM；装回 fla 后应以 --no-grad-ckpt 重跑对照。')
    if info['train_json'] and info['loss'] == 'ce':
        info['notes'].append(
            'loss=ce 在导出 JSON 数据面上 = assistant 应答段（最终符号版是裸符号，'
            '旧导出是 Answer: …；两者都含 <|im_end|>）的逐 token CE（answer-only），'
            '与 Vision-Jev 的 stage-1 形式一致。')
    if info['loss'] == 'combined':
        info['notes'].append(
            'loss=combined：choice=0.5*CE+0.5*可接受集质量+0.1*Brier'
            '（无 accept_mask 时质量项退化为 CE）；noul=BCE；score=NLL+0.5*RPS。')
    info['organ_crop'] = args.organ_crop
    info['crop_margin'] = float(args.crop_margin)
    info['crop_cache_n'] = int(args.crop_cache_n)
    if (args.organ_crop != 'off' or info['organ_evidence'] or info['organ_route']):
        info['organ_map'] = organ_map_info
    if args.organ_crop != 'off':
        crop_desc = {
            'z': '按 mask bbox 的 z 范围选层（平面全幅）',
            'bbox': '按 mask bbox 的 z 范围选层 + 平面裁到器官并集 bbox',
            'mask': '按 stitched mask 的逐层器官体素剖面选层（平面全幅）',
        }[args.organ_crop]
        info['notes'].append(
            f'器官裁剪臂（--organ-crop {args.organ_crop}）：{crop_desc}；对照=同参数 off。'
            '读数为「器官先验（+高分辨率）」的合成增益，**不是**模型自主定位；无映射/组无'
            '类目/卷缺定位列逐题回退 off 文件并计数（slice_stats.json）。mask 臂要求缓存'
            '有 organ_zprof 列，旧缓存缺列逐题回退 z 取层并计 mask_noprof。FUSE 与 '
            'ORGAN_CROP 各自立项 —— 对照臂必须同参数')
    if sfrontend is not None:
        info['slice_frontend'] = sfrontend.describe()
        info['radar_slot_id'] = int(ph_id)
        if info['organ_evidence']:
            info['ev_slot_id'] = int(ev_ph_id)
        info['freeze_vision'] = bool(args.freeze_vision)
        if info['pure2d']:
            info['notes'].append(
                f'纯 2D 切片通道：每题 {sfrontend.n_slices} 张切片走宿主原生塔（来自 RADAR '
                '那一路预处理后的体积：同一 ROI、同一窗）；序列里没有 RADAR 器官槽，'
                '第 37 槽/整图支路也不存在')
        else:
            info['notes'].append(
                f'B 支：每题 {sfrontend.n_slices} 张切片走宿主原生塔（来自 RADAR 那一路预处理后的'
                '体积：同一 ROI、同一窗），器官槽另占空闲 id —— 两条通道的消融必须各自立项，'
                '原生塔看到的是被 anisotropic resize 过的图，与自然图像预训练分布差得远')
    if getattr(args, 'fuse', False):
        info['notes'].append(
            '融合臂（--fuse）：切片 token (Q) 对 36/37 器官向量 (K/V) 的交叉注意力，'
            '零初始化 ⇒ 初始恒等。与 FUSE=0 的对照必须同 rules/split/步数，'
            '不然就是换了条臂在比')
    if info['fusion_pool']:
        info['notes'].append(
            f'器官条件池化（--fusion-pool {info["fusion_pool"]}）：{info["fusion_pool"]} 个'
            '器官 query 把全部切片 token（8 片 × ~143 = ~1,144）压成同数量的摘要 token，'
            '序列里不再有逐片图像占位符 —— 3D token 占比从 ~3% 抬到与器官槽相当。'
            '整图槽不产生摘要（它不是器官），仍按原样进序列。摘要走 1D 位置（不再算 '
            'M-RoPE，切片网格与摘要数对不上）；两跳（--fuse）的对照 = 同参数不池化'
            '（--fusion-pool 0），单跳（不带 --fuse）的对照 = 同参数不池化的裸切片臂或'
            '题绑定证据 token 臂（--organ-evidence），按所立的那一格选。')
    if info['organ2d']:
        info['notes'].append(
            '逐器官 2D 聚合臂（--organ2d）：切片 token 不进序列（序列保持 36/37 器官槽的 '
            'direct3d 结构），宿主 2D 塔输出按 organ_box 空间归属聚到器官粒度，经零初始化'
            '门控残差（proj 无 bias + 逐槽 α=0）注入 projector 输出 —— 初始态与 organ-only '
            '逐位相等，训练后 α 的分布直接回答「2D 有没有用」。归属是几何的（organ_box），'
            '不是学习式注意力（--fusion-pool 实测 AUC≈0.5 的反面）。缺 organ_box 的卷逐题'
            '回退 organ-only 并计数（slice_stats.json 的 o2d_no_box）。对照臂 = 同参数 '
            '不 --organ2d。')
    if info['organ_evidence']:
        info['notes'].append(
            '器官区域证据 token 臂（--organ-evidence）：题绑定器官（organ_map）的 organ_box '
            '区域切片 token 经零初始化 cross-att 得一个证据 token，交错拼在对应器官槽之后'
            '（ev 占位用第二个空闲 id）—— 初始态证据 = 器官向量本身（拼接不是残差，器官槽'
            '不动）。区域空/缺 organ_box 的卷证据 token 退回器官向量并计数（slice_stats.json '
            '的 oev_no_*）。无绑定器官的题回退整图全局槽（第 37 槽，行号 n_organs）：'
            'query=整图 projector 嵌入、K/V=该样本全部切片 token（无 box 限制），计数 '
            'oev_whole_fallback；编码器无整图槽（has_whole=False）时无处回退，维持 '
            'organ-only 并计数 oev_no_whole_slot。与 --fuse/--fusion-pool/--organ2d 互斥。'
            '对照臂 = 同参数不 --organ-evidence。')
    if info.get('evidence_no_box'):
        info['notes'].append(
            '证据臂的**无几何归属**消融（--evidence-no-box）：证据 token 的个数、位置、'
            '交错方式、参数量、序列形状全部与 --organ-evidence 一致，唯一改动是绑定器官'
            '的 query 不再只看 organ_box 区域内的切片 token，而是看该样本**全部**切片 token'
            '（与整图回退槽同一份 K/V，差别只剩 query）。零初始化 ⇒ step 0 与旗舰逐位相等，'
            '差异从第一次更新起。它回答的是「旗舰那 0.8330 里，box 的几何归属占多少，'
            '多一组可训 cross-att 本身占多少」—— 无归属的注意力正是 --fusion-pool 实测 '
            'AUC≈0.5 的死因，所以这一格是有先验理由怀疑、仍要跑的对照。附带一处真差异：'
            '默认臂里区域为空/缺 organ_box 的卷退回器官向量本身（计 oev_no_box），本臂'
            '没有这个退路，那批题拿到的是真证据 token。对照臂 = 同参数 --organ-evidence '
            '不 --evidence-no-box。')
    if info['organ_route']:
        if info['organ_route_visual']:
            info['notes'].append(
                '题绑定器官**每器官视觉条件**软门控臂（--organ-route-visual）：把标量 gamma '
                '换成每器官一个权重 a_k = bias_k + proj_k·v_k/√D，再走同一个均值居中门控 '
                'scale=1+tanh(a_k)·(m−mean m)，并对 Δ 再做一次样本内去均值（36 个 scale 均值'
                '恒 1）。bias_k 是器官静态先验，proj_k·v_k 让同一器官在不同图上拿到不同权重；'
                'bias/proj 零初始化 ⇒ step 0 与不开路由逐位相等，且全 bias 相等、proj=0 时'
                '严格退化为标量版。整图第 37 槽仍中性；无绑定题（route_mask 全 False）恒等、'
                '退回 organ-only 并计数（ort_no_binding/ort_ok）。与所有臂正交，不要求 '
                '--slices>0。对照臂 = 同参数 --organ-route（标量版）或不开路由。')
        elif info['organ_route_perorgan']:
            info['notes'].append(
                '题绑定器官**每器官静态标量**软门控臂（--organ-route-perorgan）：把标量 gamma '
                '换成 36 个各管各的 g_k=tanh(bias_k)，走同一个均值居中门控 '
                'scale_k=1+tanh(bias_k)·(m_k−mean m)，并对 Δ 再做一次样本内去均值（36 个 scale '
                '均值恒 1）。权重是静态的（不看视觉特征），与 --organ-route-visual 的区别就在'
                '这里；全 bias 相等时严格退化为标量版。bias 零初始化 ⇒ step 0 与不开路由逐位'
                '相等；整图第 37 槽仍中性；无绑定题（route_mask 全 False）恒等、退回 '
                'organ-only 并计数（ort_no_binding/ort_ok）。与所有臂正交，不要求 --slices>0。'
                '对照臂 = 同参数 --organ-route（标量版）或不 --organ-route。')
        else:
            info['notes'].append(
                '题绑定器官软门控臂（--organ-route）：按 organ_map 对绑定器官槽做均值居中的'
                '乘性门控（scale=1+tanh(gamma)·(m−mean m)，绑定槽上调、其余下调、整图第 37 槽'
                '中性），零初始化 gamma=0 ⇒ 初始态与 organ-only 逐位相等，训练后 gamma 的取值'
                '直接回答「路由有没有用」。无绑定的题（route_mask 全 False）门控恒等、退回 '
                'organ-only 并计数（slice_stats.json 的 ort_no_binding；可路由题计 ort_ok）。'
                '与所有臂正交可组合（不要求 --slices>0，direct3d 也成立）。对照臂 = 同参数'
                '不 --organ-route。')
    if info['organ_slots'] == 'off' and not info['pure2d']:
        info['notes'].append(
            '器官槽关闭（--organ-slots off，ClinFusion 原样）：序列里没有器官占位槽，3D '
            '特征只以融合 K/V 身份进入模型 —— projector 不进计算图（对齐段训的是融合），'
            '对照臂 = 同参数 --organ-slots on；两臂的对照必须同 rules/split/步数/切片数')
    if info['pure2d']:
        info['notes'].append(
            '纯 2D 宿主臂（--pure2d）：feats=none，RADAR 特征/器官槽/融合全部不读不进；'
            '视觉唯一入口是宿主原生塔的切片（--slices 片/题）。projector 冻结且不进 ckpt；'
            '这是直接全参微调，没有对齐段。对照臂 = 同宿主同切片数、但接 3D 特征/器官槽的臂；'
            '读数只走 score_exact() 的逐题准确率')
    if args.feats == 'zeros':
        info['notes'].append('feats=zeros 是冒烟模式（零嵌入 + has_feat 全 False）—— 只验流水线，不许报数')
    if args.feats == 'live':
        info['notes'].append(
            'feats=live：冻结塔在训练进程里现算（三级取数 RAM->npz->现算，见 visual_frontend.py）。'
            '实测单卡一个 epoch 全现算 ≈ 42 h，write-through 的 npz 是唯一的续跑凭据')
        if frontend is not None and frontend.identity == 'radar_ft':
            info['notes'].append(
                '编码器 radar_ft 见过 the source CT training cohort（≈ 我们的 train+val）：它的特征可用，'
                '但 ② 级地板在这一支是「同域微调的冻结读出」，不能当第三方佐证')
    if '+' in args.train_split:
        info['notes'].append(
            f'train_split={args.train_split}：合并集 = **没有 held-out 选择集**。后果逐条认：'
            '① 步数/epoch 必须预登记，不许按验证集挑 checkpoint；② 逐题先验地板（majority）'
            '在训练分布上拟合，只能在 test 上读；③ 唯一可报的数是 test。理由：radar_ft 见过 '
            '≈train+val，val 上的选择从来不是无污染的，留着它只等于少一份训练数据')
    if args.tiny:
        info['notes'].append('--tiny：随机小权重，只验形状与数学')
    if args.freeze_llm:
        info['notes'].append(
            '--freeze-llm：只训 projector/organ_emb/missing —— 这是**对齐段**。它的产物不许当 ③ 报数'
            '（LLM 从未被训过，指标只反映「projector 出来的坐标能不能被读」）；真正的 ③ 是从这里 '
            '--init-from 接续、解冻 LLM 的那一次')
    if getattr(args, 'init_from', None):
        info['init_from'] = args.init_from
        info['notes'].append(
            f'--init-from {args.init_from}：继承对齐段的 projector 权重，优化器/步数全新'
            f'（核对见 init_from.json）')
    if extra:
        info.update(extra)
    return info


def _check_disjoint(train_split, eval_split, who):
    """合并 train+val 之后，评估集与训练集重叠是**静默的**错误（`--split val` 以前是 held-out，
    现在是训练数据）。这里当场拒，不给「在训练集上报数」留任何路径。"""
    ts, es = set(train_split.split('+')), set(eval_split.split('+'))
    if ts & es:
        raise SystemExit(
            f'{who}：评估集 {sorted(eval_split.split("+"))} 与训练集 {sorted(ts)} 重叠 '
            f'（交集 {sorted(ts & es)}）—— 2026-10-12 起 train+val 合并为训练集，'
            f'在它上面算的指标是训练集指标。评估只能用 test')


def _dump_visual(frontend, out, suffix=''):
    """把前端的三级命中计数落盘（现算了多少例、跳过了多少例是结果的一部分，不是日志噪声）。

    `suffix` 只给分片 eval 用（片各写自己的 `visual_stats_shard{i}_of_{n}.json`，
    四片同写一个文件就是互相盖）；单卡路径传空串，文件名与落点逐字不变。
    """
    if frontend is None:
        return
    d = dict(frontend.describe())
    d['stats'] = dict(frontend.stats)
    _write_json(os.path.join(out, f'visual_stats{suffix}.json'), d)
    print(f'[visual] {frontend.identity} stats={d["stats"]}', flush=True)


def _dump_slice_stats(sfrontend, ds, out, suffix=''):
    """切片前端的取数与裁剪计数落盘（fallback 逐题计数是结果的一部分，不是日志噪声）。

    `sfrontend is None`（direct3d 的 SLICES=0）在有 dataset 计数时**也要落**：题绑定路由
    的 ort_ok/ort_no_binding 是 dataset 侧的逐题账，那条臂没有别的地方记它。没有计数就
    照旧不产文件（既有 direct3d 臂的产物目录不会因为这次改动多出一个 json）。
    """
    stats = dict(ds.organ_stats) if ds is not None else {}
    if sfrontend is None and not stats:
        return
    d = dict(sfrontend.describe()) if sfrontend is not None else {'sfrontend': None}
    d['organ_stats'] = stats
    _write_json(os.path.join(out, f'slice_stats{suffix}.json'), d)
    print(f'[slice] {d.get("stats", {})} organ_stats={d["organ_stats"]}', flush=True)


def sys_argv():
    return list(sys.argv)


def _init_distributed():
    """torchrun 启动时初始化 NCCL 进程组。返回 (rank, world_size, local_rank)。
    非分布式环境（单卡 / CPU）返回 (0, 1, 0)，后续代码按 world_size==1 走原路径。

    timeout：torch 对 NCCL 默认 600s，而 save_every 处「rank0 落 ckpt、其余卡在
    barrier 里等」这一段就要吃满写盘时间（2B ckpt 13.3 GB、4B 27.3 GB，NAS 实测
    314 MB/s ⇒ 42–87s；容器内更慢）。步数已由 StudyShuffledSampler 补齐拉平，
    这里的超时只需要容得下一次写盘，取 3600s（NCCL_TIMEOUT_S 可覆盖）。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        from datetime import timedelta
        to = int(os.environ.get('NCCL_TIMEOUT_S', '3600'))
        dist.init_process_group(backend='nccl', timeout=timedelta(seconds=to))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _load_pi_table(path, who):
    """`prefix → π_q`（该题在 train+val 里众数标签的占比）。**只由训练标签算**。

    产物 `out_logs/prior_per_qid.tsv` 由 `out_logs/pi_strata.py --regen` 从
    `out_dataset_final_sym/train_val.json` 生成，全程不读 test 标签 ⇒ 拿它当校准目标
    不构成 oracle-on-test（与「在 test 上拟合 TS」那件事是两回事）。
    """
    import csv
    if not os.path.isfile(path):
        raise SystemExit(f'{who}: π_q 表不存在 —— {path}\n'
                         '    生成：python3 out_logs/pi_strata.py --regen（只读 train_val.json）')
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter='\t'):
            out[r['qid']] = float(r['pi_mode'])
    if not out:
        raise SystemExit(f'{who}: π_q 表是空的 {path}')
    return out


def _pi_table_sha(path):
    """π 表内容的短哈希，进 run_info。理由与 organ_map 的 sha 同一条：表换了内容而路径没换，
    两条臂就不是同一个 π 条件化口径，事后从 argv 完全看不出来。"""
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def _racd_prior_check(pi, ds, mode, lam, lo=0.55, hi=0.85):
    """π 表的覆盖率硬闸。返回表（或 None），缺题就停，不静默降级。

    为什么必须停：查不到的题在 collate 里记 nan ⇒ 带判据把它算成「带外」⇒ 低权，
    于是 `cal_mode=pi`（armH）塌回成「全卷小权」= armG 的弱化版，两条臂的差别
    只剩名字，而损失照算、曲线照降、eval 指标照出。那种「配置没生效但一切正常」
    是本项目吃过最多次的一类错（见 losses.py:175 的两次 m_rank 教训）。
    λ_cal=0 时整段不生效，返回 None ⇒ 既有臂逐位不变。
    """
    if lam <= 0:
        return None
    qs = {ex[2] for ex in ds.examples}
    miss = sorted(qs - set(pi))
    if miss:
        raise SystemExit(
            f'--racd-prior-table 覆盖不全：训练集 {len(qs)} 题里有 {len(miss)} 题查不到 π_q，'
            f'例如 {miss[:5]}。\n'
            '    重建：python3 out_logs/pi_strata.py --regen（只读 train_val.json，不碰 test）。\n'
            '    不许静默补 0：π=0 会被带判据当成「低先验题」，权重方向整个反掉。')
    if mode == 'pi':
        v = [pi[q] for q in qs]
        inb = sum(1 for x in v if lo <= x <= hi)
        print(f'[racd-cal] π 表覆盖 {len(qs)}/{len(qs)} 题；带内 [{lo},{hi}] {inb} 题 '
              f'= {100.0 * inb / len(qs):.1f}%', flush=True)
    return pi


def _racd_lam_cal(args, step):
    """两阶段点火的当步 λ_cal：前 `warm` 步恒 0（纯 armF 口径），随后 `warm` 步线性升到
    目标值。理由见 `losses.racd_loss` docstring 的 cal 段 —— 可靠性项与 CE 在同一份
    logits 上符号相反（CE 要 p_max→1，这一项要 p_max→E[y]），第 1 步就开会把早期对抗
    写进权重；700 步预算里后一半 LR 本来就在退火，点火放在退火段开头最省。
    λ_cal=0（缺省）时整段短路，既有臂一个浮点都不动。"""
    tgt = float(args.racd_lam_cal)
    if tgt <= 0:
        return 0.0
    w = int(args.racd_cal_warm)
    if w <= 0:
        return tgt
    if step < w:
        return 0.0
    return tgt * min(1.0, (step - w) / float(w))


def cmd_train(args):
    if args.feats is None:
        raise SystemExit('--feats 必须显式给（它决定特征从哪来，也决定 run_info 里能报什么）：\n'
                         '  live  —— 冻结塔在训练进程里现算（2026-10-12 的口径：作为一个 VLM 跑）\n'
                         '  cache —— 读 stage 1 预提取的 npz（pretrain 那支已跑完 25,484 例）\n'
                         '  zeros —— 冒烟，只验流水线，不许报数')
    if args.feats == 'none' and not getattr(args, 'pure2d', False):
        raise SystemExit('--feats none 只配 --pure2d：不读 3D 特征时器官槽/融合都没有入口')
    if getattr(args, 'pure2d', False) and args.feats != 'none':
        raise SystemExit(f'--pure2d 要 --feats none（收到 {args.feats}）—— 纯 2D 不读任何 '
                         '3D 特征；缓存/前端/零填充三条路都不该被打开')
    _check_json_pair(args, '[train]')
    rank, world_size, local_rank = _init_distributed()
    is_main = (rank == 0)
    tok, tok_path = _load_tokenizer(args)
    dev = f'cuda:{local_rank}' if world_size > 1 else _dev(args)
    frontend = _build_frontend(args, dev)
    model, llm_ident = _build_model(args, tok, frontend)
    sfrontend, ph_id, ev_ph_id = _native_channel(args, tok, model)
    out = _mkdir(args.out)
    organ_map = organ_map_info = None
    if (args.organ_crop != 'off' or getattr(args, 'organ_evidence', False)
            or getattr(args, 'organ_route', False)):
        organ_map, om_sha, om_n = _load_organ_map(args.organ_map, need=True)
        organ_map_info = {'path': os.path.abspath(args.organ_map), 'sha256': om_sha,
                          'n_rows': om_n}
    train_ds = _dataset(tok, args.train_split, args, frontend=frontend,
                        sfrontend=sfrontend, ph_id=ph_id, organ_map=organ_map,
                        ev_ph_id=ev_ph_id)
    _check_rows(model, train_ds)
    racd = getattr(args, 'loss', 'combined') == 'racd'
    grpo = getattr(args, 'loss', 'combined') == 'grpo'
    dpo = getattr(args, 'loss', 'combined') == 'dpo'
    evgrpo = getattr(args, 'loss', 'combined') == 'evgrpo'
    if racd:
        _check_racd(args, train_ds)
    if grpo:
        _check_grpo(args, train_ds)
    if dpo:
        _check_dpo(args, train_ds)
    if evgrpo:
        _check_evgrpo(args, train_ds)
    mv = racd or dpo or (grpo and (args.grpo_lam_corr > 0 or args.grpo_balance == 'group'))
    racd_info = grpo_info = dpo_info = evgrpo_info = None
    racd_pi_sha = (_pi_table_sha(args.racd_prior_table)
                   if (racd and args.racd_prior_table) else None)
    if evgrpo:
        from .views import GroupStratifiedSampler, AcquireDataset, load_organ_bindings
        bindings, b_sha, b_n = load_organ_bindings(args.organ_map)
        train_ds = AcquireDataset(train_ds, bindings, n_samples=args.ev_samples)
        _skw = dict(seed=args.seed, bs=args.bs, rank=rank, world_size=world_size,
                    tau=args.racd_tau, min_frac=args.racd_min_frac,
                    max_repeat=args.racd_max_repeat)
        _sinfo = {'tau': float(args.racd_tau), 'min_frac': float(args.racd_min_frac),
                  'max_repeat': int(args.racd_max_repeat)}
        if args.grpo_balance == 'group':
            if args.grpo_kind_alpha < 1.0:
                if getattr(train_ds, 'kind_of', None) is None:
                    raise SystemExit('--grpo-kind-alpha<1 要逐题的题型表，但这份数据没有 —— '
                                     '只有走 --train-json 的入口才会建 `_json_qspace`')
                from .views import KindBalancedSampler
                sampler = KindBalancedSampler(train_ds, alpha=args.grpo_kind_alpha, **_skw)
                samp_info = {'kind': 'group_stratified+kind_balanced',
                             'kind_alpha': float(args.grpo_kind_alpha),
                             'kind_rows': sampler.kind_rows,
                             'kind_targets': sampler.targets,
                             'quotas': sampler.quotas(), 'group_rows': sampler.sizes,
                             **_sinfo}
            else:
                sampler = GroupStratifiedSampler(train_ds, **_skw)
                samp_info = {'kind': 'group_stratified', 'quotas': sampler.quotas(),
                             'group_rows': sampler.sizes, **_sinfo}
        else:
            sampler = StudyShuffledSampler(train_ds.study_slices, seed=args.seed, shuffle=True,
                                           rank=rank, world_size=world_size)
            samp_info = {'kind': 'study_shuffled'}
        bind_info = {'path': os.path.abspath(args.organ_map), 'sha256': b_sha,
                     'n_rows': b_n}
        evgrpo_info = _evgrpo_info(args, samp_info, bind_info)
        if is_main:
            print(f'[evgrpo] G={args.ev_samples} 子集/题 λ_cost={args.ev_lam_cost} '
                  f'β_cal={args.ev_beta_cal} β_KL={args.ev_beta_kl} '
                  f'λ_task={args.ev_lam_task}；balance={args.grpo_balance} '
                  f'分层 tau={args.racd_tau} min_frac={args.racd_min_frac} '
                  f'max_repeat={args.racd_max_repeat}', flush=True)
            if 'quotas' in samp_info:
                q = samp_info['quotas']
                small = sorted(q.items(), key=lambda kv: kv[1])[:3]
                print('[evgrpo] 组配额（最小三组）：'
                      + ' '.join(f'{k}={v}' for k, v in small)
                      + f'；epoch 长度={len(sampler)}', flush=True)
    elif mv:
        from .views import GroupStratifiedSampler, MultiViewDataset, load_organ_bindings
        bindings, b_sha, b_n = load_organ_bindings(args.organ_map)
        if args.racd_blind and grpo and args.grpo_view_weight != 'grounded':
            raise SystemExit('--racd-blind 对 --loss racd 有定义；grpo 段只有 '
                             '--grpo-view-weight grounded 用得到盲视图（作先验基线）。'
                             '要在 RL 段用先验基线，请加 --grpo-view-weight grounded')
        train_ds = MultiViewDataset(
            train_ds, bindings, n_views=args.racd_views,
            seed=args.seed, mask_whole=args.racd_mask_whole,
            blind=args.racd_blind,
            prior=(_racd_prior_check(
                       _load_pi_table(args.racd_prior_table, '--racd-prior-table'),
                       train_ds, args.racd_cal_mode, args.racd_lam_cal,
                       args.racd_cal_pi_lo, args.racd_cal_pi_hi)
                   if args.racd_lam_cal > 0 else None))
        if args.racd_mask_whole and train_ds.whole_row is None:
            raise SystemExit(f'--racd-mask-whole 开了，但这条臂没有整图槽可遮：'
                             f'n_vis={getattr(train_ds, "n_vis", "?")} != '
                             f'n_organs+1={int(getattr(train_ds, "n_organs", 0)) + 1}'
                             f'（whole_feat 没进缓存，或这条臂的 RADAR 槽根本不进序列）')
        _skw = dict(seed=args.seed, bs=args.bs, rank=rank, world_size=world_size,
                    tau=args.racd_tau, min_frac=args.racd_min_frac,
                    max_repeat=args.racd_max_repeat)
        _sinfo = {'tau': float(args.racd_tau), 'min_frac': float(args.racd_min_frac),
                  'max_repeat': int(args.racd_max_repeat)}
        if grpo and args.grpo_kind_alpha < 1.0:
            if getattr(train_ds, 'kind_of', None) is None:
                raise SystemExit('--grpo-kind-alpha<1 要逐题的题型表，但这份数据没有 —— '
                                 '只有走 --train-json 的入口才会建 `_json_qspace`')
            from .views import KindBalancedSampler
            sampler = KindBalancedSampler(train_ds, alpha=args.grpo_kind_alpha, **_skw)
            samp_info = {'kind': 'group_stratified+kind_balanced',
                         'kind_alpha': float(args.grpo_kind_alpha),
                         'kind_rows': sampler.kind_rows,
                         'kind_targets': sampler.targets,
                         'quotas': sampler.quotas(), 'group_rows': sampler.sizes,
                         **_sinfo}
        else:
            sampler = GroupStratifiedSampler(train_ds, **_skw)
            samp_info = {'kind': 'group_stratified', 'quotas': sampler.quotas(),
                         'group_rows': sampler.sizes, **_sinfo}
        bind_info = {'path': os.path.abspath(args.organ_map), 'sha256': b_sha,
                     'n_rows': b_n}
        if racd:
            racd_info = {
                'stage': 1, 'n_views': int(args.racd_views),
                'mask_whole': train_ds.whole_row is not None,
                'lam_rank': float(args.racd_lam_rank), 'lam_null': float(args.racd_lam_null),
                'm_rank': float(args.racd_m_rank), 'm_null': float(args.racd_m_null),
                'task_views': args.racd_task_views,
                'blind': bool(train_ds.blind),
                'lam_blind': float(args.racd_lam_blind), 'm_blind': float(args.racd_m_blind),
                'lam_cal': float(args.racd_lam_cal), 'cal_mode': str(args.racd_cal_mode),
                'cal_warm': int(args.racd_cal_warm),
                'cal_pi': [float(args.racd_cal_pi_lo), float(args.racd_cal_pi_hi),
                           float(args.racd_cal_pi_w)],
                'prior_table': (str(args.racd_prior_table) if args.racd_prior_table else None),
                'prior_sha': racd_pi_sha,
                'sampler': samp_info, 'bindings': bind_info,
            }
        if grpo:
            grpo_info = _grpo_info(args, samp_info, bind_info, args.racd_views,
                                   train_ds.whole_row is not None)
        if dpo:
            dpo_info = {
                'stage': 2, 'n_views': int(args.racd_views),
                'mask_whole': train_ds.whole_row is not None,
                'blind': bool(train_ds.blind),
                'beta': float(args.dpo_beta), 'lam_ce': float(args.dpo_lam_ce),
                'rejected_view': 'full',
                'ref': (os.path.abspath(args.dpo_ref or args.init_from)
                        if args.dpo_beta > 0 else None),
                'sampler': samp_info, 'bindings': bind_info,
            }
        if is_main:
            tag = 'racd' if racd else ('grpo' if grpo else 'dpo')
            mw = '是' if train_ds.whole_row is not None else '否'
            if racd:
                print(f'[racd] M={args.racd_views} 视图/题 λ_rank={args.racd_lam_rank} '
                      f'λ_null={args.racd_lam_null} m_rank={args.racd_m_rank} '
                      f'm_null={args.racd_m_null}；遮整图槽={mw}；'
                      f'答案视图={args.racd_task_views}；'
                      f'盲视图={"是" if train_ds.blind else "否"}'
                      f'（λ_blind={args.racd_lam_blind} m_blind={args.racd_m_blind}）；'
                      f'分层 tau={args.racd_tau} '
                      f'min_frac={args.racd_min_frac} max_repeat={args.racd_max_repeat}',
                      flush=True)
            elif grpo:
                ka = (f' 题型 alpha={args.grpo_kind_alpha}（'
                      + '/'.join(f'{k}={v}' for k, v in
                                 samp_info.get('kind_targets', {}).items())
                      + ' 行/epoch）' if args.grpo_kind_alpha < 1.0
                      else f' 题型 alpha={args.grpo_kind_alpha}（按行数，现状）')
                print(f'[grpo] G={args.grpo_samples} rollout/题 λ_rank={args.grpo_lam_rank} '
                      f'm_rank={args.grpo_m_rank} λ_corr={args.grpo_lam_corr} '
                      f'm_corr={args.grpo_m_corr} α={args.grpo_alpha} '
                      f'β_KL={args.grpo_beta} λ_task={args.grpo_lam_task} '
                      f'w_task={args.grpo_w_task}；'
                      f'M={args.racd_views} 视图/题 遮整图槽={mw} 分层 tau={args.racd_tau} '
                      f'min_frac={args.racd_min_frac} max_repeat={args.racd_max_repeat}'
                      + ka,
                      flush=True)
            else:
                print(f'[dpo] 跨视图硬负例 DPO（不采样）β={args.dpo_beta} '
                      f'λ_ce={args.dpo_lam_ce}；chosen=gold/rejected=blind 先验答案，'
                      f'均在 full 视图取 logπ；M={args.racd_views} 视图/题 遮整图槽={mw} '
                      f'盲视图={"是" if train_ds.blind else "否"}；分层 tau={args.racd_tau} '
                      f'min_frac={args.racd_min_frac} max_repeat={args.racd_max_repeat}',
                      flush=True)
            q = samp_info['quotas']
            small = sorted(q.items(), key=lambda kv: kv[1])[:3]
            print(f'[{tag}] 组配额（最小三组）：'
                  + ' '.join(f'{k}={v}' for k, v in small)
                  + f'；epoch 长度={len(sampler)}', flush=True)
    else:
        if grpo:
            grpo_info = _grpo_info(args, {'kind': 'study_shuffled'}, None, 1)
        sampler = StudyShuffledSampler(train_ds.study_slices, seed=args.seed, shuffle=True,
                                       rank=rank, world_size=world_size)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, sampler=sampler,
                                         collate_fn=train_ds.collate, num_workers=0)
    info = _run_info(args, tok_path, model, llm_ident, {args.train_split: train_ds.describe()},
                     frontend=frontend, sfrontend=sfrontend, ph_id=ph_id,
                     organ_map_info=organ_map_info, ev_ph_id=ev_ph_id)
    if racd_info is not None:
        info['racd'] = racd_info
    if grpo_info is not None:
        info['grpo'] = grpo_info
    if dpo_info is not None:
        info['dpo'] = dpo_info
    if evgrpo_info is not None:
        info['evgrpo'] = evgrpo_info
    if is_main:
        _write_json(os.path.join(out, 'run_info.json'), info)
        print(f'[train] out={out} mode={args.mode} rules={args.rules} '
              f'n_examples={len(train_ds)} bs={args.bs} accum={args.accum} '
              f'lr={args.lr} dtype={args.dtype} slices={args.slices} '
              f'loss={getattr(args, "loss", "ce")} '
              f'data={"json" if args.train_json else "internal"} tiny={args.tiny} '
              f'ngpu={world_size}', flush=True)
        if grpo and args.grpo_lam_corr > 0.35:
            print(f'[grpo] ⚠ λ_corr={args.grpo_lam_corr} > 0.35：RAC §3.3 在 λ_corr=0.4 '
                  f'实测 acc 0.656→0.589（塌方），其最优是 0.3。越过这条线的读数不能'
                  f'直接和论文对齐', flush=True)
        if grpo and args.grpo_w_task == 0 and args.grpo_lam_task == 0 and args.grpo_beta == 0:
            print('[grpo] ⚠ w_task=0 且 λ_task=0 且 β_KL=0：没有任何一项在锚住准确率，'
                  'pg 只剩 r^corr 一股反锐化力 ⇒ 预期 π 压平到欠自信。'
                  '要「只改校准、指标不掉」就得给 λ_task>0（proper score）或 β>0（KL 锚）',
                  flush=True)
        if grpo and args.grpo_w_task != 1.0:
            print(f'[grpo] Eq.11 自核口径（w_task={args.grpo_w_task}）：r_mean = '
                  f'{args.grpo_w_task}·r_task + {args.grpo_lam_rank}·r_rank + '
                  f'{args.grpo_lam_corr}·r_corr。r_task 读数记的是**未加权**的原始奖励均值'
                  f'（跨臂可比），所以 run_rl.sh 提示里那句不带 w_task 的等式在本臂不成立',
                  flush=True)
        if args.train_json:
            obj = ('单 token 候选集受限打分（combined）'
                   if getattr(args, 'loss', 'ce') == 'combined' else
                   'assistant 应答段 answer-only token CE')
            print(f'[train] 数据面 {args.train_json}：题面逐字取自导出文件，{obj}', flush=True)

    model.to(dev)
    params = model.trainable_parameters()
    n_tr = sum(p.numel() for p in params)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    sched = None
    resume_sched = None
    if args.lr_sched == 'cosine':
        tot = int(args.lr_total or args.limit or 0)
        if tot <= 0:
            raise SystemExit('--lr-sched cosine 要一个衰减终点：给 --lr-total <总步数>'
                             '（或 --limit，它本身就是总步数）。没有终点就没有 T_max，'
                             '跑过头之后 CosineAnnealingLR 会按周期把 lr 升回去 —— '
                             '那是一条没预注册过的臂')
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tot, eta_min=0.0)
    step, start_epoch = 0, 0
    resume_skip = 0
    ck_last = os.path.join(out, 'ckpt_last.pt')
    if args.init_from and args.resume not in ('auto', 'off'):
        raise SystemExit('--init-from 与 --resume <路径> 不合：前者是换一段训练（优化器全新），'
                         '后者是同一次跑的延续。要接续就先别带 --resume')
    if args.init_from:
        src = _init_from(model, args.init_from, args, llm_ident)
        if is_main:
            if args.resume == 'auto' and os.path.isfile(ck_last):
                print(f'[train] 注意：--out {out} 里已有 ckpt_last.pt，但带了 --init-from —— '
                      f'本次不续跑它（续跑与「从对齐段接续」互斥）。想续跑本次请换个 --out', flush=True)
            _write_json(os.path.join(out, 'init_from.json'), src)
    elif args.resume != 'off' and os.path.isfile(args.resume if args.resume != 'auto' else ck_last):
        p = args.resume if args.resume != 'auto' else ck_last
        ck = torch.load(p, map_location='cpu', weights_only=False)
        old_json = (ck.get('guard') or {}).get('train_json')
        now_json = getattr(args, 'train_json', None)
        if old_json != now_json:
            raise SystemExit(f'--resume {p} 的训练数据面是 {old_json}，本次是 {now_json} '
                             '—— 数据面不同不是同一次跑；换 --out 从头起跑')
        old_rules = set(str(ck.get('rules') or '').split(',')) - {''}
        now_rules = set(args.rules.split(',')) - {''}
        if old_rules != now_rules:
            raise SystemExit(f'--resume {p} 的 --rules={sorted(old_rules)}，本次={sorted(now_rules)} '
                             '—— 题集不同不是同一次跑；换 --out 从头起跑')
        old_pnc = bool((ck.get('guard') or {}).get('prompt_no_count', False))
        if old_pnc != bool(getattr(args, 'prompt_no_count', False)):
            raise SystemExit(f'--resume {p} 的题面数量词口径是 no_count={old_pnc}，'
                             f'本次是 {bool(getattr(args, "prompt_no_count", False))} —— '
                             '口径不同不是同一次跑；换 --out 从头起跑，或续旧臂时显式 '
                             'PROMPT_NO_COUNT=0')
        old_sched = ck.get('sched') is not None
        if old_sched != (args.lr_sched == 'cosine'):
            raise SystemExit(f'--resume {p} 的 LR 调度是 '
                             f'{"cosine" if old_sched else "const"}，本次是 {args.lr_sched} '
                             '—— 调度不同不是同一次跑；换 --out 从头起跑')
        model.load_trainable_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        resume_sched = ck.get('sched')
        step, start_epoch = ck['step'], ck['epoch']
        if 'micro_in_epoch' in ck:
            resume_skip = int(ck['micro_in_epoch']) * args.bs
        elif start_epoch == 0:
            resume_skip = int(step) * args.accum * args.bs
        if is_main:
            print(f'[train] 续跑 {p}：epoch={start_epoch} step={step}'
                  + (f'；快进已见数据 {resume_skip} 题' if resume_skip else ''), flush=True)
    else:
        if is_main:
            print(f'[train] 干净起跑，可训参数 {n_tr / 1e6:.2f}M（LLM frozen={args.freeze_llm}）', flush=True)
    if sched is not None and resume_sched is not None:
        sched.load_state_dict(resume_sched)
        for g, lr_now in zip(opt.param_groups, sched.get_last_lr()):
            g['lr'] = lr_now
        if is_main:
            print(f'[train] LR 调度续接：cosine T_max={sched.T_max} '
                  f'last_epoch={sched.last_epoch} lr={opt.param_groups[0]["lr"]:.3e}',
                  flush=True)
    step0 = step

    ref_model = None
    if (grpo and args.grpo_beta > 0) or (dpo and args.dpo_beta > 0) \
            or (evgrpo and args.ev_beta_kl > 0):
        _ref_path = args.grpo_ref if grpo else (args.dpo_ref if dpo else args.ev_ref)
        _ref_beta = args.grpo_beta if grpo else (args.dpo_beta if dpo else args.ev_beta_kl)
        ref_model = copy.deepcopy(model)
        if _ref_path:
            _init_from(ref_model, _ref_path, args, llm_ident)
        ref_model.requires_grad_(False)
        ref_model.eval()
        if is_main:
            _rtag = 'grpo' if grpo else ('dpo' if dpo else 'evgrpo')
            print(f'[{_rtag}] 参考策略 = '
                  f'{os.path.abspath(_ref_path or args.init_from)}'
                  f'；β={_ref_beta}（已冻结 + eval）', flush=True)

    if args.grad_ckpt:
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        if is_main:
            print('[train] grad_ckpt=on：逐层激活反向重算（无 fla 的线性注意力兜底必需）',
                  flush=True)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    raw_model = model.module if hasattr(model, 'module') else model
    log = os.path.join(out, 'train_log.jsonl')
    diag_num, diag_den = collections.defaultdict(float), collections.defaultdict(float)
    if racd:
        from .losses import RACD_DIAG_W as DIAG_W
    elif grpo:
        from .losses import GRPO_DIAG_W as DIAG_W
    elif dpo:
        from .losses import DPO_DIAG_W as DIAG_W
    elif evgrpo:
        from .losses import EVGRPO_DIAG_W as DIAG_W
    else:
        DIAG_W = {}
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        skip = resume_skip if epoch == start_epoch else 0
        sampler.set_skip(skip)
        micro_base = skip // max(1, args.bs)
        if skip and is_main:
            print(f'[train] 快进 {skip} 题（epoch {epoch} 已见数据），从断点继续', flush=True)
        resume_skip = 0
        model.train()
        run_loss, run_n, micro, broke = 0.0, 0, 0, False
        run_drop, run_cand, run_tok = 0, 0, 0
        diag_num.clear()
        diag_den.clear()
        for batch in loader:
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            ctx = (torch.autocast('cuda', dtype=torch.bfloat16)
                   if dev.startswith('cuda') and args.dtype == 'bfloat16' else _nullctx())
            with ctx:
                token_ce = bool(args.train_json) and args.loss == 'ce'
                out_f = None if evgrpo else model(batch, token_ce=token_ce)
                if getattr(args, 'loss', 'ce') == 'combined':
                    from .losses import combined_loss
                    loss_raw = combined_loss(
                        out_f['scores'], batch['target'],
                        [k[0] for k in batch['space_key']],
                        accept_mask=batch.get('accept_mask'))
                elif racd:
                    from .losses import racd_loss
                    loss_raw, racd_diag = racd_loss(
                        out_f['scores'], batch['target'],
                        [k[0] for k in batch['space_key']], batch['view_kind'],
                        n_views=batch['n_views'],
                        lam_rank=args.racd_lam_rank, lam_null=args.racd_lam_null,
                        m_rank=args.racd_m_rank, m_null=args.racd_m_null,
                        accept_mask=batch.get('accept_mask'),
                        task_views=args.racd_task_views,
                        lam_blind=args.racd_lam_blind, m_blind=args.racd_m_blind,
                        lam_cal=_racd_lam_cal(args, step),
                        cal_mode=args.racd_cal_mode,
                        cal_prior=(batch.get('prior')
                                   if args.racd_lam_cal > 0 and args.racd_cal_mode == 'pi'
                                   else None),
                        cal_pi_lo=args.racd_cal_pi_lo, cal_pi_hi=args.racd_cal_pi_hi,
                        cal_pi_w=args.racd_cal_pi_w)
                elif grpo:
                    from .losses import grpo_loss
                    ref_s = None
                    if ref_model is not None:
                        with torch.no_grad():
                            ref_s = ref_model(batch, token_ce=False)['scores']
                    loss_raw, grpo_diag = grpo_loss(
                        out_f['scores'], batch['target'],
                        [k[0] for k in batch['space_key']],
                        accept_mask=batch.get('accept_mask'),
                        ref_scores=ref_s,
                        view_kind=batch.get('view_kind'),
                        n_views=int(batch.get('n_views', 0) or 0),
                        n_samples=args.grpo_samples, beta=args.grpo_beta,
                        lam_task=args.grpo_lam_task,
                        w_task=args.grpo_w_task,
                        lam_rank=args.grpo_lam_rank, m_rank=args.grpo_m_rank,
                        lam_corr=args.grpo_lam_corr, m_corr=args.grpo_m_corr,
                        alpha_sev=args.grpo_alpha,
                        gate_corr=not args.grpo_no_gate_corr,
                        corr_cand=args.grpo_corr_cand,
                        task_views=args.grpo_task_views,
                        view_weight=args.grpo_view_weight)
                elif dpo:
                    from .losses import dpo_loss
                    ref_s = None
                    if ref_model is not None:
                        with torch.no_grad():
                            ref_s = ref_model(batch, token_ce=False)['scores']
                    loss_raw, dpo_diag = dpo_loss(
                        out_f['scores'], batch['target'],
                        [k[0] for k in batch['space_key']], batch['view_kind'],
                        n_views=int(batch['n_views']),
                        ref_scores=ref_s,
                        beta=args.dpo_beta, lam_ce=args.dpo_lam_ce,
                        accept_mask=batch.get('accept_mask'))
                elif evgrpo:
                    from .losses import evgrpo_loss
                    from .views import apply_acq_mask
                    G = int(args.ev_samples)
                    n_organs = int(raw_model.n_organs)
                    gen = torch.Generator(device=dev)
                    gen.manual_seed(int(args.seed) + 1000003 * epoch
                                    + 10007 * (micro_base + micro) + rank)
                    out_f = model(batch, evgrpo={'n_samples': G, 'n_organs': n_organs,
                                                 'generator': gen})
                    ref_s = None
                    if ref_model is not None:
                        masked = apply_acq_mask(dict(batch), out_f['subsets'], n_organs)
                        with torch.no_grad():
                            ref_s = ref_model(masked, token_ce=False)['scores']
                    tgt_q = batch['target'][::G]
                    kinds_q = [batch['space_key'][i][0]
                               for i in range(0, len(batch['space_key']), G)]
                    accept_q = (batch['accept_mask'][::G]
                                if batch.get('accept_mask') is not None else None)
                    route_q = (batch['route_mask'][::G]
                               if batch.get('route_mask') is not None else None)
                    loss_raw, evgrpo_diag = evgrpo_loss(
                        out_f['acq_logits'], out_f['subsets'], out_f['present'],
                        out_f['scores'], tgt_q, kinds_q, accept_mask=accept_q,
                        ref_scores=ref_s, route_mask=route_q,
                        n_samples=G, lam_cost=args.ev_lam_cost,
                        beta_cal=args.ev_beta_cal, beta_kl=args.ev_beta_kl,
                        lam_task=args.ev_lam_task)
                else:
                    loss_raw = out_f['loss']
                loss = loss_raw / args.accum
            loss.backward()
            run_loss += loss_raw.item()
            run_n += 1
            if racd or grpo or dpo or evgrpo:
                _accum_diag(racd_diag if racd else (grpo_diag if grpo else
                            (dpo_diag if dpo else evgrpo_diag)),
                            diag_num, diag_den, DIAG_W)
            if 'scores' in out_f:
                run_drop += int(out_f['n_dropped'])
                run_cand += int(out_f['scores'].numel())
            else:
                run_tok += int(out_f['n_tok'])
            micro += 1
            if micro % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if sched is not None:
                    sched.step()
                if args.log_every and step % args.log_every == 0 and is_main:
                    el = time.time() - t0
                    rec = {'epoch': epoch, 'step': step, 'loss': run_loss / run_n,
                           'sec': round(el, 1),
                           'sec_per_step': round(el / max(1, step - step0), 3)}
                    if sched is not None:
                        rec['lr'] = opt.param_groups[0]['lr']
                    if run_cand:
                        rec['drop_frac'] = round(run_drop / max(1, run_cand), 4)
                    if run_tok:
                        rec['n_tok'] = run_tok
                    if frontend is not None:
                        rec['visual_stats'] = dict(frontend.stats)
                    extra = (f'drop={run_drop}/{run_cand}' if run_cand
                             else f'tok={run_tok}')
                    if (racd or grpo or dpo or evgrpo) and diag_den:
                        rd = {k: round(diag_num[k] / diag_den[k], 6)
                              for k in diag_num if diag_den[k]}
                        if racd:
                            dt, dd = rd.get('d_tgt', 0.0), rd.get('d_dist', 0.0)
                            rd['asym'] = round(dt - dd, 6)
                            rd['ratio'] = round(dt / dd, 3) if abs(dd) > 1e-6 else None
                            rec['racd'] = rd
                            extra += (f' Δtgt={dt:+.4f} Δdist={dd:+.4f} '
                                      f'gate={rd.get("gate_frac", 0.0):.3f} '
                                      f'keep={rd.get("task_keep", 1.0):.3f}')
                        elif grpo:
                            rec['grpo'] = rd
                            extra += _grpo_extra(rd)
                        elif evgrpo:
                            rec['evgrpo'] = rd
                            extra += _evgrpo_extra(rd)
                        else:
                            rec['dpo'] = rd
                            extra += _dpo_extra(rd)
                    print(f'[train] ep{epoch} step{step} loss={run_loss / run_n:.4f} '
                          f'{extra} {el:.0f}s', flush=True)
                    _log_jsonl(log, rec)
                    run_loss, run_n = 0.0, 0
                    run_drop, run_cand, run_tok = 0, 0, 0
                    if racd or grpo or dpo or evgrpo:
                        diag_num.clear()
                        diag_den.clear()
                if args.save_every and step % args.save_every == 0:
                    if world_size > 1:
                        dist.barrier()
                    if is_main:
                        _save(ck_last, raw_model, opt, step, epoch, args, llm_ident,
                              micro_in_epoch=micro_base + micro, sched=sched)
                    if world_size > 1:
                        dist.barrier()
            if args.limit and step >= args.limit:
                broke = True
                break
        if not broke and micro % args.accum:
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if sched is not None:
                sched.step()
        if world_size > 1:
            dist.barrier()
        if is_main:
            _save(ck_last, raw_model, opt, step, epoch + (0 if broke else 1), args, llm_ident,
                  micro_in_epoch=micro_base + micro, sched=sched)
            _dump_visual(frontend, out)
            _dump_slice_stats(sfrontend, train_ds, out)
        if world_size > 1:
            dist.barrier()
        if broke:
            if is_main:
                print(f'[train] epoch {epoch} 被 --limit={args.limit} 中断，step={step}，ckpt={ck_last}',
                      flush=True)
            break
        if is_main:
            print(f'[train] epoch {epoch} 完，step={step}，ckpt={ck_last}', flush=True)
    if is_main:
        print(f'[train] 结束：step={step} 用时 {time.time() - t0:.0f}s', flush=True)
        if getattr(args, 'organ_route', False) and getattr(raw_model, 'route', None) is not None:
            rs = raw_model.route.summary()
            info['model']['route_summary'] = rs
            if rs.get('mode') == 'scalar':
                info['model']['route_gamma'] = rs['gamma']
                print(f'[train] route_gamma={rs["gamma"]:+.4f}（写回 run_info.model）',
                      flush=True)
            elif rs.get('mode') == 'visual':
                b = rs['bias']
                info['model']['route_bias'] = b
                info['model']['route_proj_norm'] = rs['proj_norm']
                print(f'[train] route_visual proj_norm={rs["proj_norm"]:.4f} '
                      f'bias[min/mean/max]={min(b):+.4f}/{sum(b) / len(b):+.4f}/{max(b):+.4f}'
                      '（写回 run_info.model）', flush=True)
            else:
                b = rs['bias']
                info['model']['route_bias'] = b
                info['model']['route_gamma_mean'] = rs['gamma_mean']
                info['model']['route_gamma_std'] = rs['gamma_std']
                info['model']['route_gamma_min'] = rs['gamma_min']
                info['model']['route_gamma_max'] = rs['gamma_max']
                print(f'[train] route_perorgan gamma[min/mean/max]='
                      f'{rs["gamma_min"]:+.4f}/{rs["gamma_mean"]:+.4f}/'
                      f'{rs["gamma_max"]:+.4f}（写回 run_info.model）', flush=True)
            _write_json(os.path.join(out, 'run_info.json'), info)
        _dump_visual(frontend, out)
        _dump_slice_stats(sfrontend, train_ds, out)
    if world_size > 1:
        dist.destroy_process_group()
    return 0


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _save(path, model, opt, step, epoch, args, llm_ident=None, micro_in_epoch=0, sched=None):
    tmp = path + '.tmp'
    torch.save({'model': model.trainable_state_dict(), 'opt': opt.state_dict(),
                'step': step, 'epoch': epoch,
                'micro_in_epoch': int(micro_in_epoch),
                'sched': sched.state_dict() if sched is not None else None,
                'rules': args.rules, 'mode': args.mode,
                'guard': {'llm': llm_ident, 'n_tok': model.n_tok,
                          'encoder': args.encoder, 'train_split': args.train_split,
                          'slices': int(getattr(args, 'slices', 0) or 0),
                          'train_json': getattr(args, 'train_json', None),
                          'feats': args.feats,
                          'fuse': bool(getattr(args, 'fuse', False)),
                          'fusion_pool': int(getattr(args, 'fusion_pool', 0) or 0),
                          'organ_crop': args.organ_crop,
                          'crop_margin': float(args.crop_margin),
                          'crop_cache_n': int(args.crop_cache_n),
                          'organ_slots': str(getattr(args, 'organ_slots', 'on')),
                          'organ2d': bool(getattr(args, 'organ2d', False)),
                          'organ_evidence': bool(getattr(args, 'organ_evidence', False)),
                          'evidence_no_box': bool(getattr(args, 'organ_evidence', False)
                                                  and getattr(args, 'evidence_no_box', False)),
                          'organ_route': bool(getattr(args, 'organ_route', False)),
                          'organ_route_visual': bool(getattr(args, 'organ_route_visual',
                                                             False)),
                          'organ_route_perorgan': bool(getattr(
                              args, 'organ_route_perorgan', False)),
                          'organ_acquire': bool(getattr(args, 'organ_acquire', False)),
                          'prompt_no_count': bool(getattr(args, 'prompt_no_count',
                                                           False)),
                          'vis_tokens': str(getattr(args, 'vis_tokens', 'all') or 'all'),
                          'organ_index': str(getattr(args, 'organ_index', 'on')),
                          'slot_perm': int(getattr(args, 'slot_perm', 0) or 0)},
               'loss': str(getattr(args, 'loss', 'ce'))},
               tmp)
    os.replace(tmp, path)


def _init_from(model, path, args, llm_ident):
    """两阶段训练的第一段产物 -> **只取权重**（优化器与步数全新）。

    与 `--resume` 的分工：resume = 同一次跑的延续（连 optimizer/step 一起接）；
    init-from = 换一段训练（典型：projector-only 对齐跑完，解冻 LLM 接着练）。
    """
    ck = torch.load(path, map_location='cpu', weights_only=False)
    g = ck.get('guard') or {}
    if not g:
        raise SystemExit(f'{path} 里没有 guard 块 —— 它是加这个字段之前存的，'
                         f'无法核对身份；要用它请手工确认 LLM/编码器后用 --resume')
    if ck.get('mode') == 'state-only':
        raise SystemExit(f'{path} 是 state-only（n_vis=0）跑出来的：projector 从未进过计算图，'
                         f'那份权重就是随机初始化，接它等于没接')
    same = {'LLM': (g.get('llm') or {}).get('config_sha256'), 'n_tok': g.get('n_tok'),
            '编码器': g.get('encoder'), '训练集': g.get('train_split'),
            '切片数': g.get('slices'),
            '训练数据面': g.get('train_json'),
            '融合': bool(g.get('fuse', False)),
            '池化': int(g.get('fusion_pool', 0) or 0),
            '器官裁剪': '{}:{}:{}'.format(g.get('organ_crop', 'off'),
                                         g.get('crop_margin', 0.1),
                                         g.get('crop_cache_n', 24)),
            '器官槽': str(g.get('organ_slots', 'on')),
            'organ2d': bool(g.get('organ2d', False)),
            'organ_evidence': bool(g.get('organ_evidence', False)),
            'evidence_no_box': bool(g.get('evidence_no_box', False)),
            'organ_route': bool(g.get('organ_route', False)),
            'organ_route_visual': bool(g.get('organ_route_visual', False)),
            'organ_route_perorgan': bool(g.get('organ_route_perorgan', False)),
            '题面数量词': bool(g.get('prompt_no_count', False)),
            'vis_tokens': str(g.get('vis_tokens', 'all')),
            'organ_index': str(g.get('organ_index', 'on')),
            'slot_perm': int(g.get('slot_perm', 0) or 0)}
    now = {'LLM': (llm_ident or {}).get('config_sha256'), 'n_tok': model.n_tok,
           '编码器': args.encoder, '训练集': args.train_split,
           '切片数': int(getattr(args, 'slices', 0) or 0),
           '训练数据面': getattr(args, 'train_json', None),
           '融合': bool(getattr(args, 'fuse', False)),
           '池化': int(getattr(args, 'fusion_pool', 0) or 0),
           '器官裁剪': '{}:{}:{}'.format(args.organ_crop, args.crop_margin,
                                        args.crop_cache_n),
           '器官槽': str(getattr(args, 'organ_slots', 'on')),
           'organ2d': bool(getattr(args, 'organ2d', False)),
           'organ_evidence': bool(getattr(args, 'organ_evidence', False)),
           'evidence_no_box': bool(getattr(args, 'organ_evidence', False)
                                   and getattr(args, 'evidence_no_box', False)),
           'organ_route': bool(getattr(args, 'organ_route', False)),
           'organ_route_visual': bool(getattr(args, 'organ_route_visual', False)),
           'organ_route_perorgan': bool(getattr(args, 'organ_route_perorgan', False)),
           '题面数量词': bool(getattr(args, 'prompt_no_count', False)),
           'vis_tokens': str(getattr(args, 'vis_tokens', 'all') or 'all'),
           'organ_index': str(getattr(args, 'organ_index', 'on')),
           'slot_perm': int(getattr(args, 'slot_perm', 0) or 0)}
    diff = [k for k in same
            if not (k == '切片数' and same[k] is None) and same[k] != now[k]]
    if diff:
        raise SystemExit(f'--init-from {path} 与本次不一致：' + '；'.join(
            f'{k} {same[k]} → {now[k]}' for k in diff) +
            ' —— 对齐段与全参段必须是同一个 LLM、同一支编码器、同一个训练集，'
            '否则继承来的 projector 是对着别的东西校准的坐标')
    old_rules = set(str(ck.get('rules') or '').split(',')) - {''}
    now_rules = set(args.rules.split(',')) - {''}
    if old_rules != now_rules:
        raise SystemExit(
            f'--init-from {path} 的 --rules={sorted(old_rules)}，本次={sorted(now_rules)} '
            '—— 对齐段与全参段必须用同一份题集（输出目录名不带 rules）；'
            '换规则 = 换一条臂，从它自己的对齐段接续')
    model.load_trainable_state_dict(ck['model'])
    n_proj = sum(1 for k in ck['model'] if k.startswith('projector.'))
    n_fuse = sum(1 for k in ck['model'] if k.startswith('fusion.'))
    print(f'[train] init-from {path}：继承 {len(ck["model"])} 个张量（projector {n_proj} 个，'
          f'融合 {n_fuse} 个，来源 step={ck.get("step")}），优化器与步数全新', flush=True)
    return {'path': os.path.abspath(path), 'from_step': ck.get('step'),
            'from_mode': ck.get('mode'), 'n_tensors': len(ck['model']),
            'guard': g}


def _verdict_line(study_id, qid, kind, y_key, y_idx, p_key, p_idx, p_pos, p_dist):
    """analysis/verdict_metrics.py 冻结输入 schema 的**一行**逐题判定。

    两处写盘共用这一份：`_predict` 边推理边落的流式文件，与 eval 末尾按 `metrics.collect`
    行表写的权威全量文件。分叉就会让同一批产物有两种 schema，而复算脚本只认一种。
    `p_key=None` = 未覆盖（只有权威文件会出现，流式文件按定义只含真推过的题）。
    """
    if p_key is None:
        gold = y_idx if kind != 'choice' else y_key
        parsed = probs = raw = None
        ok = False
    elif kind == 'noul':
        gold, parsed, raw = y_idx, p_idx, p_key
        probs = ({'a': 1.0 - float(p_pos), 'b': float(p_pos)}
                 if p_pos is not None else None)
        ok = p_key == y_key
    elif kind == 'choice':
        gold, parsed, raw = y_key, p_key, p_key
        probs = p_dist
        ok = p_key == y_key
    else:
        gold, parsed, raw = y_idx, p_idx, p_key
        probs = ({str(j + 1): float(p) for j, p in enumerate(p_dist)}
                 if p_dist is not None else None)
        ok = p_key == y_key
    return {'study_id': str(study_id), 'question_id': qid, 'type': kind,
            'gold': gold, 'parsed': parsed, 'raw': raw,
            'probs': probs, 'ok': ok}


@torch.no_grad()
def _predict(model, ds, *, bs, limit, dev, log_every=200, vpath=None, rows=None,
             oneshot=1):
    """prompt-only 数据集 -> (pred, p_pos, p_dist)。分数走 `score_exact()`（报数只走这条）。

    `p_dist` 收 score 题（档位顺序下的概率向量，给 MAE 做概率加权）和 choice 题
    （候选键 -> 概率，给 verdict_metrics 的疾病级 mAUC）。

    `rows` = 要推的行号序列（默认整卷）。多卡 eval 用它把卷切成连续块：
    `--num-shards n --shard i` 各跑 `[i·ceil(N/n), min(N,(i+1)·ceil(N/n)))`，互不重叠。
    给了 `rows` 才套 `Subset`，不给就逐字是原来的整卷路径。

    `vpath` 给了就**边推理边落**逐题判定（每 batch flush 一次，与 mllm_jev_infer 的
    流式口径一致）。动机：概率质量是唯一无法事后复算的东西 —— acc/bal_acc/auroc 之类
    汇总反解不出 Σp²，一旦某臂只留了聚合 metrics，Brier/mAUC 这些列就永久缺（direct3D
    与 OrganRoute 的 noul 侧就是这样丢的，要补只能重上 GPU）。gold 直接取
    `batch['target']`（dataset.__getitem__ 里的 `pack.target_index`），不额外推一遍。
    """
    idxs = None if rows is None else list(rows)
    span = len(ds) if idxs is None else len(idxs)
    sub = ds if idxs is None or span == len(ds) else torch.utils.data.Subset(ds, idxs)
    loader = torch.utils.data.DataLoader(sub, batch_size=bs, shuffle=False,
                                         collate_fn=ds.collate, num_workers=0)
    pred = collections.defaultdict(dict)
    ppos = collections.defaultdict(dict)
    pdist = collections.defaultdict(dict)
    model.eval()
    done = 0
    n_v = 0
    vf = open(vpath, 'w', encoding='utf-8') if vpath else None
    try:
        for batch in loader:
            tgts = batch['target'].tolist()
            batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if getattr(model, 'acquire', None) is not None:
                from .views import apply_acq_mask
                with torch.no_grad():
                    acq_logits = model.acquire_scores(batch)
                    present = batch['has_feat'].bool()
                    map_sub = ((acq_logits > 0) & present).unsqueeze(1)
                    batch = apply_acq_mask(batch, map_sub, int(model.n_organs))
            scores, cvalid = model.score_exact(batch, oneshot=bool(oneshot))
            sel = scores.masked_fill(~cvalid, float('-inf')).argmax(dim=1)
            probs = torch.softmax(scores.masked_fill(~cvalid, float('-inf')), dim=1)
            for r in range(len(batch['cand_ids'])):
                q, i = batch['question'][r], str(batch['study_id'][r])
                space = ds.space_of(batch['space_key'][r])
                n_k = len(space.keys)
                p_idx = int(sel[r])
                p_key = space.keys[p_idx]
                pred[q][i] = p_key
                if space.kind == 'noul':
                    p_pos, p_dist = float(probs[r, 1]), None
                    ppos[q][i] = p_pos
                elif space.kind == 'choice':
                    p_pos = None
                    p_dist = {space.keys[j]: float(probs[r, j]) for j in range(n_k)}
                    pdist[q][i] = p_dist
                else:
                    p_pos = None
                    p_dist = [float(x) for x in probs[r, :n_k]]
                    pdist[q][i] = p_dist
                if vf is not None:
                    y_idx = int(tgts[r])
                    vf.write(json.dumps(_verdict_line(
                        i, q, space.kind, space.keys[y_idx], y_idx,
                        p_key, p_idx, p_pos, p_dist), ensure_ascii=False) + '\n')
                    n_v += 1
            if vf is not None:
                vf.flush()
            done += len(batch['cand_ids'])
            if log_every and done % (log_every * bs) < bs:
                print(f'[eval] {done}/{limit or span}', flush=True)
            if limit and done >= limit:
                break
    finally:
        if vf is not None:
            vf.close()
            print(f'[eval] 流式逐题判定 {vpath}（{n_v} 行，每推一批就落盘）', flush=True)
    return pred, ppos, pdist


def _shard_bounds(n_total, num_shards, shard):
    """连续块切分：片 i 拿 `[lo,hi)`。切**连续块**而不是隔行取，是为了让每个 shard 的
    DataLoader 批次与整卷单卡跑时同构（同题的候选集相邻、切片缓存 LRU 命中形状不变），
    否则 fps 与单卡不可比，也就没法用单卡计时外推多卡预算。"""
    if num_shards < 1:
        raise SystemExit(f'--num-shards={num_shards} < 1')
    if not (0 <= shard < num_shards):
        raise SystemExit(f'--shard={shard} 不在 [0,{num_shards})')
    per = -(-n_total // num_shards)
    lo = min(shard * per, n_total)
    return lo, min(lo + per, n_total)


def _shard_paths(out, shard, num_shards):
    """片的两份产物。verdict 那份**故意叫 `verdicts_shard*of*.jsonl`**：
    `analysis/verdict_metrics.load_rows` 就 glob `verdicts_shard*.jsonl`，
    ⇒ 分片 eval 一落盘，冻结校准口径直接能读（不必等 merge），而 `verdicts_stream`
    那个名字它认不出，半截产物不会被静默当全量读数。"""
    tag = f'{shard}_of_{num_shards}'
    return (os.path.join(out, f'verdicts_shard{tag}.jsonl'),
            os.path.join(out, f'eval_shard{tag}.json'))


def _load_shards(out, n_total):
    """读全部 `eval_shard*_of_*.json` -> (pred, ppos, pdist)。校验覆盖与不重叠。

    merge 只并字典（`pred` 是 `{question: {study_id: key}}`，按名字索引不是按位置），
    所以各片并行跑完直接并集；真正要核的是**行覆盖**：块必须恰好铺满 [0,n_total)，
    少一片或多一片都当场 SystemExit —— 静默按缺片的分母出报告，就是拿一个更小的
    test 集报数，而报告里看不出来。
    """
    ps = sorted(glob.glob(os.path.join(out, 'eval_shard*_of_*.json')))
    if not ps:
        raise SystemExit(f'{out} 里没有 eval_shard*_of_*.json —— 分片没跑过？')
    metas, pred, ppos, pdist = [], {}, {}, {}
    idents = set()
    for p in ps:
        d = json.load(open(p, encoding='utf-8'))
        metas.append({k: d[k] for k in ('shard', 'num_shards', 'lo', 'hi', 'n_rows',
                                        'seconds', 'rows_per_s')})
        metas[-1]['option_perm'] = int(d.get('option_perm', 0) or 0)
        idents.add((d.get('split'), d.get('ckpt'), metas[-1]['option_perm']))
        if d['hi'] - d['lo'] != d['n_rows']:
            raise SystemExit(f'片 {d["shard"]} 的 n_rows={d["n_rows"]} 与块宽 '
                             f'{d["hi"]}-{d["lo"]} 不符')
        for src, dst in (('pred', pred), ('ppos', ppos), ('pdist', pdist)):
            for q, per_study in d[src].items():
                slot = dst.setdefault(q, {})
                for sid, v in per_study.items():
                    if sid in slot:
                        raise SystemExit(f'片 {d["shard"]} 与先前的片在 {q}/{sid} 上重叠')
                    slot[sid] = v
    n = len(metas)
    declared = sorted({m['num_shards'] for m in metas})
    if n != metas[0]['num_shards'] or len(declared) > 1:
        raise SystemExit(f'片不齐或混了两次分片跑：目录里 {n} 个 eval_shard*_of_*.json，'
                         f'声明的 N 有 {declared} —— 要么补跑缺的片，要么把上一次分片的'
                         f' eval_shard*/verdicts_shard* 产物移走再跑这一次（不能混）')
    ks = sorted(m['shard'] for m in metas)
    if ks != list(range(n)):
        raise SystemExit(f'片号不是一组 0..{n - 1}：{ks} —— 同一片跑过两遍，或编号写错')
    if len(idents) > 1:
        raise SystemExit(f'片来自不同的 split/ckpt：{sorted(idents)} —— 一个报告不许混两条 run')
    blocks = sorted((m['lo'], m['hi']) for m in metas)
    cur = 0
    for lo, hi in blocks:
        if lo != cur:
            raise SystemExit(f'行覆盖不连续：期望 lo={cur}，实到 lo={lo}（块 {blocks}）')
        cur = hi
    if cur != n_total:
        raise SystemExit(f'片覆盖到 {cur} 行，但 split 有 {n_total} 行 —— 不许按缺片报数')
    return pred, ppos, pdist, metas


def _write_verdicts(rows, path):
    """metrics.collect 的行表 -> verdict_metrics.py 冻结口径的逐题判定（权威全量文件）。

    训练 eval 原先只落聚合报告，choice 的疾病级 mBAcc/mAUC 与 overall 的 56 病加权
    值无从复算。这里按 analysis/verdict_metrics.py 的输入 schema 落盘：
    noul gold/parsed=0/1、probs={'a':P(No),'b':P(Yes)}；choice gold/parsed=候选键
    （符号版是字母）、probs=候选键->概率；score gold/parsed=0 基档位、probs=1 基数字键。
    行内容由 `_verdict_line` 定，与 `_predict` 的流式文件同一份构造器。
    """
    n = 0
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            (study_id, qid, _prefix, kind, y_key, y_idx, p_key, p_idx,
             _ordinal, p_pos, _rule, p_dist) = r
            f.write(json.dumps(_verdict_line(study_id, qid, kind, y_key, y_idx,
                                             p_key, p_idx, p_pos, p_dist),
                               ensure_ascii=False) + '\n')
            n += 1
    return n


def _check_stream(path, rows):
    """流式文件 vs 末尾权威行表：同键同行、逐字段相等。

    两条 gold 来源是独立的 —— 流式取 `batch['target']`（dataset.__getitem__ 算的），
    权威行由 `metrics.collect` 从 `ds.examples` 重推 `pack.target_index`。对上才说明
    「边推理边落」那份可以当产物用（挂了之后直接拿它复算，不必重上 GPU）。
    未覆盖行（`p_key=None`）只存在于权威文件，不参与比对。

    文件名故意叫 `verdicts_stream` 而不是 `verdicts_shard*`/`verdicts_spliced`：
    `verdict_metrics.load_rows` 按名字 glob，认不出它 ⇒ 半截产物不会被静默当成全量
    读数。要复算就把它拷进一个空目录改名 `verdicts_spliced.jsonl`，并显式报覆盖题数。
    """
    want = {}
    for r in rows:
        (study_id, qid, _prefix, kind, y_key, y_idx, p_key, p_idx,
         _ordinal, p_pos, _rule, p_dist) = r
        if p_key is None:
            continue
        want[(str(study_id), qid, kind)] = _verdict_line(
            study_id, qid, kind, y_key, y_idx, p_key, p_idx, p_pos, p_dist)
    got, n_line = {}, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n_line += 1
            got[(d['study_id'], d['question_id'], d['type'])] = d
    miss = [k for k in want if k not in got]
    extra = [k for k in got if k not in want]
    diff = [k for k in want if k in got and got[k] != want[k]]
    if miss or extra or diff:
        why = diff[0] if diff else (miss[0] if miss else extra[0])
        raise SystemExit(
            f'🔴 {path} 与权威行表不一致：缺 {len(miss)} 行 / 多 {len(extra)} 行 / '
            f'值不同 {len(diff)} 行；首例 {why}\n'
            f'   流式={got.get(why)}\n   权威={want.get(why)}')
    if n_line != len(got):
        print(f'[eval] ⚠️ 流式文件有 {n_line - len(got)} 行撞键（study,question,type）'
              '，与聚合口径一致地各留一条', flush=True)
    return n_line, len(want)


def cmd_eval(args):
    run_info = os.path.join(args.run, 'run_info.json')
    if not os.path.isfile(run_info):
        raise SystemExit(f'缺 {run_info} —— --run 指向训练产物目录')
    with open(run_info, encoding='utf-8') as f:
        info = json.load(f)
    if args.rules and args.rules != ','.join(info['rules']):
        print(f"[eval] 命令行 --rules={args.rules} 被忽略 —— 判据以 run_info 为准："
              f"{','.join(info['rules'])}", flush=True)
    args.rules = ','.join(info['rules'])
    args.mode = info['mode']
    ts_run = info.get('train_split') or next(iter(info.get('data') or {}), None)
    if ts_run and ts_run != args.train_split:
        print(f'[eval] --train-split={args.train_split} 被忽略 —— 以 run_info 为准：{ts_run}',
              flush=True)
        args.train_split = ts_run
    for attr, flag in (('train_json', '--train-json'), ('eval_json', '--eval-json')):
        run_v = info.get(attr)
        cur = getattr(args, attr, None)
        if run_v:
            if cur and os.path.abspath(cur) != os.path.abspath(run_v):
                print(f'[eval] {flag}={cur} 被忽略 —— 以 run_info 为准：{run_v}', flush=True)
            setattr(args, attr, run_v)
        elif cur:
            print(f'[eval] {flag}={cur} 被忽略 —— 这次训练没有用导出 JSON（内部题面）',
                  flush=True)
            setattr(args, attr, None)
    _check_json_pair(args, '[eval]')
    _check_disjoint(args.train_split, args.split, '[eval]')
    if args.feats and args.feats != info['feats']:
        print(f"[eval] --feats={args.feats} 被忽略 —— 取数路径以 run_info 为准：{info['feats']}"
              f'（换一条路特征就不是同一份输入了）', flush=True)
    args.feats = info['feats']
    p2d_run = bool(info.get('pure2d', False))
    if bool(getattr(args, 'pure2d', False)) != p2d_run:
        print(f"[eval] --pure2d={getattr(args, 'pure2d', False)} 被忽略 —— 以 run_info 为准："
              f'{p2d_run}', flush=True)
    args.pure2d = p2d_run
    cr_run = info.get('cache_root')
    if cr_run and os.path.abspath(cr_run) != os.path.abspath(args.cache_root):
        print(f'[eval] --cache-root={args.cache_root} 被忽略 —— 以 run_info 为准：{cr_run}',
              flush=True)
        args.cache_root = cr_run
    sha_run = info.get('cache_sha12')
    if sha_run != getattr(args, 'cache_sha12', None):
        if getattr(args, 'cache_sha12', None):
            print(f'[eval] --cache-sha12={args.cache_sha12} 被忽略 —— 以 run_info 为准：'
                  f'{sha_run}', flush=True)
        args.cache_sha12 = sha_run
    if args.feats == 'live' and args.mode != 'state-only':
        vf = info.get('visual_frontend')
        if not vf:
            raise SystemExit('run_info.json 里没有 visual_frontend 块，但 feats=live —— '
                             '这份产物比前端记录早，不能评')
        for flag, attr, key in (('--encoder', 'encoder', 'encoder_identity'),
                                ('--visual-root', 'visual_root', 'root')):
            cur = getattr(args, attr)
            if cur != vf[key]:
                print(f'[eval] {flag}={cur} 被忽略 —— 以 run_info 为准：{vf[key]}', flush=True)
                setattr(args, attr, vf[key])
        args.visual_ckpt = None
    ns = int(info.get('native_slices', 0))
    if ns != args.slices:
        print(f'[eval] --slices={args.slices} 被忽略 —— 以 run_info 为准：{ns}'
              '（切片数变了就不是同一份输入）', flush=True)
        args.slices = ns
    sf_run = info.get('slice_frontend')
    fz_run = bool(info.get('freeze_vision', False))
    if sf_run is not None and args.freeze_vision != fz_run:
        print(f"[eval] --freeze-vision={args.freeze_vision} 被忽略 —— 以 run_info 为准："
              f'{fz_run}（权重 trainable 与否决定 ckpt 里存了什么）', flush=True)
        args.freeze_vision = fz_run
    fu_run = bool(info.get('fuse', False))
    if args.fuse != fu_run:
        print(f'[eval] --fuse={args.fuse} 被忽略 —— 以 run_info 为准：{fu_run}'
              '（融合模块在不在决定 ckpt 里有没有这组权重）', flush=True)
        args.fuse = fu_run
    fp_run = int(info.get('fusion_pool', 0) or 0)
    if int(getattr(args, 'fusion_pool', 0) or 0) != fp_run:
        print(f'[eval] --fusion-pool={args.fusion_pool} 被忽略 —— 以 run_info 为准：{fp_run}'
              '（摘要数决定序列结构与 ckpt 的 pool.* 权重）', flush=True)
    args.fusion_pool = fp_run
    osl = str(info.get('organ_slots', 'on'))
    if str(getattr(args, 'organ_slots', 'on')) != osl:
        print(f'[eval] --organ-slots={args.organ_slots} 被忽略 —— 以 run_info 为准：{osl}'
              '（器官槽在不在决定序列里有没有器官占位）', flush=True)
    args.organ_slots = osl
    vt_run = str(info.get('vis_tokens', 'all'))
    if str(getattr(args, 'vis_tokens', 'all')) != vt_run:
        print(f'[eval] --vis-tokens={getattr(args, "vis_tokens", "all")} 被忽略 —— '
              f'以 run_info 为准：{vt_run}（进序列的行数决定 projector 槽数与 '
              'organ_evidence 的整图回退支在不在）', flush=True)
    args.vis_tokens = vt_run
    oi_run = str(info.get('organ_index', 'on'))
    if str(getattr(args, 'organ_index', 'on')) != oi_run:
        print(f"[eval] --organ-index={getattr(args, 'organ_index', 'on')} 被忽略 —— "
              f'以 run_info 为准：{oi_run}（organ_emb.weight 与 organ_shared 是两组不同的'
              '参数名，ckpt 里只有训练时那一组）', flush=True)
    args.organ_index = oi_run
    sp_run = int(info.get('slot_perm', 0))
    if int(getattr(args, 'slot_perm', 0) or 0) != sp_run:
        print(f'[eval] --slot-perm={getattr(args, "slot_perm", 0)} 被忽略 —— 以 run_info '
              f'为准：{sp_run}（槽序是权重的属性：不复原就是拿另一个模型的输入分布评估，'
              '而它不会报错）', flush=True)
    args.slot_perm = sp_run
    o2d_run = bool(info.get('organ2d', False))
    if bool(getattr(args, 'organ2d', False)) != o2d_run:
        print(f'[eval] --organ2d={getattr(args, "organ2d", False)} 被忽略 —— 以 run_info 为准：'
              f'{o2d_run}（聚合臂在不在决定残差接线与 ckpt 的 organ2d.* 权重）', flush=True)
    args.organ2d = o2d_run
    oev_run = bool(info.get('organ_evidence', False))
    if bool(getattr(args, 'organ_evidence', False)) != oev_run:
        print(f'[eval] --organ-evidence={getattr(args, "organ_evidence", False)} 被忽略 —— '
              f'以 run_info 为准：{oev_run}（证据臂在不在决定序列结构与 ckpt 的 '
              'evidence.* 权重）', flush=True)
    args.organ_evidence = oev_run
    oa_run = bool(info.get('organ_acquire', False))
    if bool(getattr(args, 'organ_acquire', False)) != oa_run:
        print(f'[eval] --organ-acquire={getattr(args, "organ_acquire", False)} 被忽略 —— '
              f'以 run_info 为准：{oa_run}（获取头在不在决定 ev 全集与 MAP 掩码评估口径）',
              flush=True)
    args.organ_acquire = oa_run
    if oa_run and not oev_run:
        raise SystemExit('run_info.json 记了 organ_acquire=True 却没有 organ_evidence —— '
                         '产物自相矛盾，拒绝猜接线')
    evnb_run = bool(info.get('evidence_no_box', False))
    if bool(getattr(args, 'evidence_no_box', False)) != evnb_run:
        print(f'[eval] --evidence-no-box={getattr(args, "evidence_no_box", False)} 被忽略 —— '
              f'以 run_info 为准：{evnb_run}（它不改权重形状，只能靠 run_info 认臂；'
              '拿 box 路径评 -evnb 的 ckpt = 静默换臂）', flush=True)
    args.evidence_no_box = evnb_run
    ort_run = bool(info.get('organ_route', False))
    orv_run = bool(info.get('organ_route_visual', False))
    orp_run = bool(info.get('organ_route_perorgan', False))
    if orv_run and not ort_run:
        raise SystemExit('run_info.json 记了 organ_route_visual=True 却没有 organ_route —— '
                         '产物自相矛盾，拒绝猜接线')
    if orp_run and not ort_run:
        raise SystemExit('run_info.json 记了 organ_route_perorgan=True 却没有 organ_route —— '
                         '产物自相矛盾，拒绝猜接线')
    if orv_run and orp_run:
        raise SystemExit('run_info.json 同时记了 organ_route_visual 与 organ_route_perorgan —— '
                         '两套门控参数化互斥，产物自相矛盾')
    if bool(getattr(args, 'organ_route', False)) != ort_run:
        print(f'[eval] --organ-route={getattr(args, "organ_route", False)} 被忽略 —— '
              f'以 run_info 为准：{ort_run}（路由臂在不在决定器官槽缩放与 ckpt 的 '
              'route.* 权重）', flush=True)
    if bool(getattr(args, 'organ_route_visual', False)) != orv_run:
        print(f'[eval] --organ-route-visual='
              f'{getattr(args, "organ_route_visual", False)} 被忽略 —— '
              f'以 run_info 为准：{orv_run}（标量版是 route.gamma，'
              '每器官视觉版是 route.bias/route.proj，两套 ckpt 键不能混）', flush=True)
    if bool(getattr(args, 'organ_route_perorgan', False)) != orp_run:
        print(f'[eval] --organ-route-perorgan='
              f'{getattr(args, "organ_route_perorgan", False)} 被忽略 —— '
              f'以 run_info 为准：{orp_run}（每器官静态版是 route.bias，'
              '与视觉版的 route.proj 不能混）', flush=True)
    args.organ_route = ort_run or orv_run or orp_run
    args.organ_route_visual = orv_run
    args.organ_route_perorgan = orp_run
    pnc_run = bool(info.get('prompt_no_count', False))
    if bool(getattr(args, 'prompt_no_count', False)) != pnc_run:
        print(f'[eval] --prompt-no-count={getattr(args, "prompt_no_count", False)} 被忽略 —— '
              f'以 run_info 为准：{pnc_run}（题面数量词口径换一个 = 换一串输入 token）',
              flush=True)
    args.prompt_no_count = pnc_run
    op = int(getattr(args, 'option_perm', 0) or 0)
    if op and int(info.get('option_perm', 0) or 0) != op:
        print(f'[eval] --option-perm={op} 生效（run_info 记的是 '
              f'{int(info.get("option_perm", 0) or 0)}）—— 命令行说了算，'
              '这是有意的：轮转是本次评估的呈现方式，不是权重的属性', flush=True)
    args.option_perm = op
    out_want = args.out or args.run
    if op:
        for _rep in (f'report_{args.split}.md', f'metrics_{args.split}.json'):
            if os.path.exists(os.path.join(out_want, _rep)):
                raise SystemExit(
                    f'--option-perm={op} 拒绝写进 {out_want}/{_rep}（已存在）：'
                    '那份是 identity 口径（选项未轮转）的已报数产物，覆盖它就再没有对照了。'
                    '换一个全新的 --out（例如 <原目录>_perm<seed>）。')
        print(f'[eval] 🔴 --option-perm={op}：choice 题的选项描述按 sha256({op}, qid) '
              '在字母位上轮转 ⇒ 这份读数**不是 identity 口径**，'
              '不许与已报数的臂并排比大小（只能与同一 ckpt 的另一个 seed 比）。'
              'noul / score 一律不动。键是 qid 而不是 (study, qid)：同一题名在各卷上必须'
              '拿到同一个置换，否则 `qspace` 的跨卷答案空间断言会当场 FATAL'
              '（见 dataset.option_permutation 的 docstring）。', flush=True)
    if sf_run and args.slice_root != sf_run['root']:
        print(f"[eval] --slice-root={args.slice_root} 被忽略 —— 以 run_info 为准："
              f"{sf_run['root']}", flush=True)
        args.slice_root = sf_run['root']
    oc = str(info.get('organ_crop', 'off'))
    if args.organ_crop != oc:
        print(f'[eval] --organ-crop={args.organ_crop} 被忽略 —— 以 run_info 为准：{oc}', flush=True)
    args.organ_crop = oc
    args.crop_margin = float(info.get('crop_margin', 0.1))
    args.crop_cache_n = int(info.get('crop_cache_n', 24))
    organ_map = None
    if oc != 'off' or oev_run or ort_run:
        om = info.get('organ_map') or {}
        if not om.get('path'):
            raise SystemExit(f'run_info.json 记了 organ_crop={oc}/organ_evidence={oev_run}/'
                             f'organ_route={ort_run} 却没有 organ_map.path —— 产物不完整，'
                             '拒绝猜一张题-器官绑定表来评')
        organ_map, om_sha, _ = _load_organ_map(om['path'], need=True)
        if om_sha != om.get('sha256'):
            raise SystemExit(f"organ_map sha256 不符：盘上 {om['path']} 现在是 {om_sha}，"
                             f"训练时是 {om.get('sha256')} —— 命令行不许偷换题-器官绑定表")
        args.organ_map = om['path']
    llm_run = info.get('llm') or {}
    if llm_run.get('tiny'):
        if not args.tiny:
            print('[eval] 这份产物是 --tiny 的 —— 同样按 --tiny 重建（随机小初始化）',
                  flush=True)
            args.tiny = True
        if llm_run.get('processor_dir') and not args.llm:
            args.llm = llm_run['processor_dir']
            print(f"[eval] --tiny 的宿主目录取自 run_info：{args.llm}（只取 processor/tokenizer，"
                  '权重仍是随机小初始化）', flush=True)
    elif llm_run.get('path'):
        if args.llm and args.llm != llm_run['path']:
            print(f"[eval] --llm={args.llm} 被忽略 —— 以 run_info 为准：{llm_run['path']}",
                  flush=True)
        args.llm = llm_run['path']
        args.tiny = False
    tok_run = info.get('tokenizer')
    if tok_run and args.tok and args.tok != tok_run:
        print(f'[eval] --tok={args.tok} 被忽略 —— 以 run_info 为准：{tok_run}'
              '（换 tokenizer 就是换候选的 token 序列）', flush=True)
    if tok_run:
        args.tok = tok_run
    tok, _ = _load_tokenizer(args)
    dev = _dev(args)
    frontend = _build_frontend(args, dev)
    model, _ = _build_model(args, tok, frontend)
    sfrontend, ph_id, ev_ph_id = _native_channel(args, tok, model)
    if ph_id is not None and 'radar_slot_id' in info and ph_id != int(info['radar_slot_id']):
        raise SystemExit(f'器官槽占位 id {ph_id} != 训练时记的 {info["radar_slot_id"]} —— '
                         '宿主或 tokenizer 换了一套，序列里的插槽对不上')
    if ev_ph_id is not None and 'ev_slot_id' in info and ev_ph_id != int(info['ev_slot_id']):
        raise SystemExit(f'证据槽占位 id {ev_ph_id} != 训练时记的 {info["ev_slot_id"]} —— '
                         '宿主或 tokenizer 换了一套，序列里的插槽对不上')
    ck = os.path.join(args.run, 'ckpt_last.pt')
    if not os.path.isfile(ck):
        raise SystemExit(f'缺 {ck}（还没训过？）')
    st = torch.load(ck, map_location='cpu', weights_only=False)
    model.load_trainable_state_dict(st['model'])
    print(f'[eval] 载入 {ck}（step={st["step"]}）', flush=True)

    model.to(dev)
    ds = _dataset(tok, args.split, args, prompt_only=True, frontend=frontend,
                  sfrontend=sfrontend, ph_id=ph_id, organ_map=organ_map,
                  ev_ph_id=ev_ph_id)
    if getattr(args, 'eval_keep_bound', 'off') != 'off':
        from .views import KeepBoundDataset, load_organ_bindings
        bindings, b_sha, b_n = load_organ_bindings(args.organ_map)
        ds = KeepBoundDataset(ds, bindings,
                              keep_whole=args.eval_keep_bound == 'bound_whole')
        print(f'[eval] keep_bound={args.eval_keep_bound}：bindings={b_n} 行 '
              f'sha256={b_sha[:12]} {ds.describe()["keep_bound"]}', flush=True)
    _check_rows(model, ds)
    print(f'[eval] split={args.split}：{ds.describe()}', flush=True)
    out = _mkdir(args.out or args.run)
    sp = os.path.join(out, 'verdicts_stream.jsonl')
    n_eval = len(ds) if not args.limit else min(args.limit, len(ds))
    shard_metas = None
    if args.num_shards > 1 and not args.merge_shards:
        lo, hi = _shard_bounds(n_eval, args.num_shards, args.shard)
        vp_sh, jp_sh = _shard_paths(out, args.shard, args.num_shards)
        t_sh = time.time()
        pred, ppos, pdist = _predict(model, ds, bs=args.bs, limit=0, dev=dev,
                                     vpath=vp_sh, rows=range(lo, hi),
                                     oneshot=args.score_oneshot)
        sec = time.time() - t_sh
        with open(jp_sh, 'w', encoding='utf-8') as f:
            json.dump({'shard': args.shard, 'num_shards': args.num_shards,
                       'lo': lo, 'hi': hi, 'n_rows': hi - lo,
                       'seconds': round(sec, 1), 'rows_per_s': round((hi - lo) / sec, 3),
                       'split': args.split, 'ckpt': ck,
                       'option_perm': int(getattr(args, 'option_perm', 0) or 0),
                       'pred': pred, 'ppos': ppos, 'pdist': pdist}, f, ensure_ascii=False)
        print(f'[eval] 片 {args.shard}/{args.num_shards}：行 [{lo},{hi}) = {hi - lo} 行 / '
              f'{sec:.0f}s = {(hi - lo) / sec:.2f} 行/秒 -> {os.path.basename(vp_sh)}',
              flush=True)
        sfx = f'_shard{args.shard}_of_{args.num_shards}'
        _dump_visual(frontend, out, sfx)
        _dump_slice_stats(sfrontend, ds, out, sfx)
        return
    if args.merge_shards:
        pred, ppos, pdist, shard_metas = _load_shards(out, n_eval)
        op_sh = sorted({m.get('option_perm', 0) for m in shard_metas})
        if op_sh != [int(getattr(args, 'option_perm', 0) or 0)]:
            raise SystemExit(f'片的 --option-perm={op_sh} 与 merge 这一步的 '
                             f'{int(getattr(args, "option_perm", 0) or 0)} 不符 —— gold 跟着选项'
                             '轮转走，混了口径就是用一套 gold 判另一套答案。'
                             '给 merge 传与各片相同的 --option-perm。')
        tot = sum(m['n_rows'] for m in shard_metas)
        secs = sum(m.get('seconds') or 0 for m in shard_metas)
        print(f'[eval] merge {len(shard_metas)} 片 = {tot} 行（本次口径 {n_eval} 行，'
              f'split 共 {len(ds)}，覆盖已核）'
              + (f'；每卡均速 {tot / secs:.2f} 行/秒（合计 {secs:.0f}s 推理）' if secs else ''),
              flush=True)
        sp = None
    else:
        pred, ppos, pdist = _predict(model, ds, bs=args.bs, limit=args.limit, dev=dev,
                                     vpath=sp, oneshot=args.score_oneshot)
    rows = M.collect(ds, pred, ppos, pdist)
    per_q, fam, by_rule, total, by_kind = M.evaluate(rows)

    floors, floor_cmps, notes = {}, {}, []
    if op:
        notes.append(f'🔴 --option-perm={op}：choice 题的选项描述按 sha256({op}, qid) '
                     '在字母位上轮转（字母/尾行/K/cand_ids 不变，只有「哪个描述坐在哪个字母上」变；'
                     'noul 与 score 一律不动）。**本报告不是 identity 口径**，'
                     '不许与已报数的臂并排比大小 —— 只能与同一个 ckpt 的另一个 seed 比'
                     '（置换不变性闸门：flip 率与 p_gold 漂移）。')
    if args.limit:
        notes.append(f'🔴 --limit={args.limit}：本报告只覆盖前 {args.limit} 题'
                     f'（数据集顺序，全量 {len(ds)} 题）—— 不许当全集读数')
    try:
        floor_rad = FL.radar_floor(ds)
        floors['radar-cosine'] = floor_rad
        floor_cmps['radar-cosine'] = M.compare(rows, floor_rad['pred'])
        notes.append(f"② radar-cosine 地板覆盖 {floor_rad['n_covered']}/{floor_rad['n_eval']}"
                     f"（{floor_rad['uncovered_why']}），acc={_fmt(floor_rad['acc_micro_covered'])}")
    except SystemExit as e:
        notes.append(f'② radar-cosine 地板未算：{e}')
    except FileNotFoundError as e:
        notes.append(f'② radar-cosine 地板未算（缓存缺文件）：{e}')

    vp = os.path.join(out, 'verdicts_spliced.jsonl')
    n_v = _write_verdicts(rows, vp)
    if sp is None:
        tot_rows = sum(m['n_rows'] for m in shard_metas)
        tot_sec = sum(m.get('seconds') or 0.0 for m in shard_metas)
        notes.append(f'分片 merge 报告：{len(shard_metas)} 片 `verdicts_shard*_of_*.jsonl` '
                     f'并成，行覆盖已按 [0,{n_eval}) 连续铺满核过；'
                     f'每卡均速 {tot_rows / max(1e-9, tot_sec):.2f} 行/秒'
                     f'（合计 {tot_rows} 行 / {tot_sec:.0f}s），并行墙钟 ≈ '
                     f'{max(m.get("seconds") or 0.0 for m in shard_metas) / 3600:.2f} h')
        print(f'[eval] 逐题判定 {vp}（{n_v} 行，含未覆盖）；'
              f'来源 = 分片产物，无 stream 可对账', flush=True)
    else:
        n_s, n_c = _check_stream(sp, rows)
        print(f'[eval] 逐题判定 {vp}（{n_v} 行，含未覆盖）；流式 {sp} 对账 '
              f'{n_s} 行 = 权威已覆盖 {n_c} 行，逐字段一致', flush=True)
    md = os.path.join(out, f'report_{args.split}.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write(M.render_md(f"{args.split} 评估（mode={args.mode}, rules={args.rules}）",
                            per_q, fam, by_rule, by_kind, floor_cmps, notes))
    _write_json(os.path.join(out, f'metrics_{args.split}.json'),
                {'metric_scheme': 'noul/choice=acc; score=mae/qwk',
                 'option_perm': op,
                 'by_kind': by_kind,
                 'total': total, 'by_family': {f'{k[0]}/{k[1]}': v for k, v in fam.items()},
                 'by_rule': {f'{k[0]}/{k[1]}': v for k, v in by_rule.items()},
                 'per_question': {f'{k[0]}/{k[1]}': v for k, v in per_q.items()},
                 'floor_cmps': {fn: dict(c) for fn, c in floor_cmps.items()}})
    if floors:
        FL.write(os.path.join(out, 'floors.json'), *floors.values())
    if not args.merge_shards:
        _dump_visual(frontend, out)
        _dump_slice_stats(sfrontend, ds, out)
    print(f'[eval] 报告 {md}', flush=True)
    parts = []
    for kind, s in by_kind.items():
        if kind == 'score':
            parts.append(f'score MAE={_fmt(s["mae"])} QWK={_fmt(s["qwk"])}')
        else:
            parts.append(f'{kind} acc={_fmt(s["acc"])}')
    print(f'[eval] ' + ' | '.join(parts) + f'（n={total["n"]}）', flush=True)
    return 0


def _fmt(x):
    return '—' if x is None else f'{x:.4f}'


def cmd_floors(args):
    args.feats = args.feats or 'cache'
    if args.slices:
        raise SystemExit('--slices 在 floors 里没有意义：① 地板只看文本，② 地板读 npz 的'
                         '概率列，两条都不做前向 —— 要测切片通道请跑 train/eval')
    if args.organ_crop != 'off':
        raise SystemExit('--organ-crop 在 floors 里没有意义：①② 两级地板都不做前向'
                         '（切片流不存在，无物可裁）—— 要测裁剪臂请跑 train/eval')
    if getattr(args, 'organ_evidence', False):
        raise SystemExit('--organ-evidence 在 floors 里没有意义：①② 两级地板都不做前向'
                         '（证据 token 无从产生）—— 要测证据臂请跑 train/eval')
    if getattr(args, 'organ_route', False):
        raise SystemExit('--organ-route 在 floors 里没有意义：①② 两级地板都不做前向'
                         '（器官槽根本不进模型，门控无物可缩放）—— 要测路由臂请跑 train/eval')
    fp_floors = int(getattr(args, 'fusion_pool', 0) or 0)
    if fp_floors:
        raise SystemExit(f'--fusion-pool={fp_floors} 在 floors 里没有意义：①② 两级地板'
                         '都不做前向（池化的输入是切片 token）—— 要测池化臂请跑 train/eval')
    if str(getattr(args, 'organ_slots', 'on')) != 'on':
        raise SystemExit('--organ-slots off 在 floors 里没有意义：①② 两级地板都不做前向'
                         '（融合不存在，K/V 无处可去）—— 要测融合唯一臂请跑 train/eval')
    _check_json_pair(args, '[floors]')
    _check_disjoint(args.train_split, args.split, '[floors]')
    frontend = _build_frontend(args, _dev(args))
    tok, _ = _load_tokenizer(args)
    ev = _dataset(tok, args.split, args, prompt_only=True, frontend=frontend)
    tr = _dataset(tok, args.train_split, args, prompt_only=False, frontend=frontend)
    fs = [FL.majority_floor(tr, ev)]
    try:
        fs.append(FL.radar_floor(ev))
    except SystemExit as e:
        print(f'[floors] ② 跳过：{e}', flush=True)
    out = _mkdir(args.out or '.')
    path = FL.write(os.path.join(out, f'floors_{args.split}.json'), *fs)
    for fl in fs:
        print(f"[floors] {fl['name']}: 覆盖 {fl['n_covered']}/{fl['n_eval']} "
              f"acc={_fmt(fl['acc_micro_covered'])}", flush=True)
    print(f'[floors] -> {path}', flush=True)
    return 0


def _add_common(p, *, rules_required=True):
    p.add_argument('--rules', required=rules_required,
                   help='逗号分隔的 finding 极性白名单（无隐式默认值，plan §8.1）；'
                        '取值域 ' + '/'.join(FINDING_RULES) +
                        '；不在此列的 finding 题被过滤，其余前缀不受它管。'
                        '🔴 回收来的暗示阴性挂的是 `implied_negative_v2`（逐绑定阈值 + '
                        '逐 item 判据两道闸，`analysis/implied_binding_recover.py`），'
                        '写 `implied_negative` 得到 0 题 —— 那一档只剩没收回的行'
                        + ('' if rules_required else
                           '。eval 不给 —— 判据以 run_info.json 为准（命令行不许偷换）'))
    p.add_argument('--mode', default='full', choices=['full', 'state-only'])
    p.add_argument('--split', default='test',
                   help='评估集：单个 split 或 "+" 合并（例 test）。合并训练集用 --train-split')
    p.add_argument('--train-split', default='train+val',
                   help='训练/地板拟合用的 split 规格。2026-10-12 用户拍板合并 train+val '
                        '（radar_ft 的 val 选择本就污染，留着只等于少一份训练数据）—— '
                        '代价见 run_info 的 notes：没有 held-out 选择集，步数必须预登记')
    p.add_argument('--feats', default=None, choices=['cache', 'live', 'zeros', 'none'],
                   help='特征来源：live=冻结塔在进程里现算（三级取数 RAM->npz->现算）；'
                        'cache=读 stage 1 的 npz；zeros=冒烟不许报数；none=纯 2D 宿主臂'
                        '（不读任何 3D 特征，必须配 --pure2d）。'
                        '不给定：train 报错要求显式选，floors 用 cache，eval 跟随 run_info')
    p.add_argument('--encoder', default='radar_ft',
                   help='--feats live 的编码器身份（radar_encoder.ENCODER_IDENTITIES 的键）：'
                        'pretrain=RADAR 原始权重 492 视觉张量/中文文本塔/无整图；'
                        'radar_ft=RADAR+ 499 张量（含 *_whole）/英文文本塔。身份决定第 37 槽是否存在')
    p.add_argument('--visual-root', default=DEFAULT_VISUAL_ROOT,
                   help='--feats live 的 write-through npz 根目录（<root>/<sha12>/<split>/<sid>.npz）')
    p.add_argument('--visual-ckpt', default=None,
                   help='不给就用身份表里的默认路径；给了也必须过 sha256 校验')
    p.add_argument('--no-write-through', action='store_true',
                   help='现算不落 npz（纯现算：单卡一个 epoch ≈ 42 h，且重启后从头再来）')
    p.add_argument('--lru', type=int, default=2048,
                   help='--feats live 的 RAM 行数上限（study 级，≈190 KB/例）')
    from .slice_frontend import DEFAULT_ROOT as _SLICE_ROOT
    p.add_argument('--slices', type=int, default=0,
                   help='B 支：每题喂给宿主原生视觉塔切片数（0=关掉，只走 RADAR 一条通道）。'
                        '宿主必须是原生多模态（--llm 指 pretrained/Qwen3.5-*），'
                        'processor 与权重同目录')
    p.add_argument('--pure2d', action='store_true',
                   help='纯 2D 宿主臂：n_vis=0，无 RADAR 特征/器官槽/融合，视觉只走宿主'
                        '原生塔（--slices>0）。必须配 --feats none --organ-slots off，'
                        '且不能配 --freeze-llm（projector 已冻，再冻 LLM 就没有可训参数）。'
                        '直接全参微调，没有对齐段')
    p.add_argument('--slice-root', default=_SLICE_ROOT,
                   help='切片缓存根目录（<root>/<split>/<N>s_<sid>.npz，存 uint8 切片而非 '
                        'pixel_values：与宿主无关，换模型不用重刷）')
    p.add_argument('--slice-lru', type=int, default=8,
                   help='切片缓存的 RAM 上限（study 级；一片 256×384 uint8 ≈ 100 KB，'
                        '默认 8 例 × --slices 片）')
    p.add_argument('--slice-proc-lru', type=int, default=0,
                   help='**processed 像素**的 RAM 备忘上限（study 级；0=关，逐位是老行为）。'
                        '上面的 --slice-lru 只省 npz 的读，命中了照样重跑宿主 processor'
                        '（32 片实测 0.70 s/卷，产物 112.5 MB）；同卷的题连着出'
                        '（test.json 每卷 11.19 题、hit_mem 91%%），这一层把整卷 eval 的 '
                        'processor 调用从 56825 压到 5080。数值逐位不变（纯函数，'
                        'analysis/check_process_determinism.py 核过 sha256）。'
                        '代价：每槽 112.5 MB(s32)/56 MB(s16)，8 槽 ≈ 900 MB/进程。'
                        '裁剪臂（--organ-crop 非 off）按题取层，命中率≈0，别开')
    p.add_argument('--freeze-vision', action='store_true',
                   help='B 支消融：冻住宿主自带的视觉塔（只留 projector/LLM 文本栈可训）。'
                        '不给就是可训 —— RADAR 那一支天生冻结（不是宿主权重），所以这个开关'
                        '只管原生通道')
    p.add_argument('--fuse', action='store_true',
                   help='B 支融合臂：切片 token（Q）对 36/37 器官向量（K/V）的交叉注意力'
                        '（ClinFusion 核心机制移植；零初始化 ⇒ 初始即恒等）。要配 --slices>0；'
                        'FUSE=0/1 是两条各自立项的臂，对照必须同 rules/split/步数')
    p.add_argument('--fusion-pool', type=int, default=0, metavar='N',
                   help='B 支器官条件池化：N 个器官 query 把全部切片 token 压成 N 个摘要，'
                        '序列里不再保留逐片图像占位符（N 必须等于器官槽数）。'
                        '摘要走 1D 位置。0 = 关（默认）；要配 --slices>0 与器官槽 on；'
                        '--fuse 可选（带上=两跳，不带=器官 query 直接读塔输出）')
    p.add_argument('--organ2d', action='store_true',
                   help='B 支逐器官 2D 聚合臂：切片 token 不进序列（序列保持 36/37 器官槽'
                        '的 direct3d 结构），宿主 2D 塔输出按 organ_box 空间归属聚到器官粒度，'
                        '经零初始化门控残差注入 projector 输出 —— 初始态与 organ-only 逐位'
                        '相等。要配 --slices>0 与器官槽 on；与 --fuse/--fusion-pool/'
                        '--organ-crop/--pure2d 互斥。对照臂 = 同参数不 --organ2d')
    p.add_argument('--organ-evidence', action='store_true',
                   help='B 支器官区域证据 token 臂：题绑定器官（--organ-map）的 organ_box '
                        '区域切片 token 经零初始化 cross-att 得一个证据 token，交错拼在'
                        '对应器官槽之后（初始态 = 器官向量本身，证据是拼接不是残差）。'
                        '要配 --slices>0 与器官槽 on；与 --fuse/--fusion-pool/--organ2d/'
                        '--organ-crop bbox/--pure2d 互斥。对照臂 = 同参数不 --organ-evidence')
    p.add_argument('--evidence-no-box', action='store_true',
                   help='证据臂的**无几何归属**消融：仍只给题绑定器官各产一个证据 token'
                        '（个数/位置/参数量/序列形状与 --organ-evidence 完全一致），但它的 '
                        'K/V 不再限定在 organ_box 区域内，而是该样本全部切片 token —— 与整图'
                        '回退槽同一份 K/V，「这个证据讲的是哪个器官」只剩 query 一个来源。'
                        '零初始化 ⇒ step 0 与旗舰逐位相等。要配 --organ-evidence；对照臂 = '
                        '同参数 --organ-evidence 不带本开关')
    p.add_argument('--organ-route', action='store_true',
                   help='题绑定器官**软门控**臂（OrganRoute，1 个标量 gamma）：按 --organ-map '
                        '把绑定器官槽 scale 上调、其余下调 —— scale[b,k] = 1 + '
                        'tanh(gamma)·(m[b,k] − mean_k m[b,k])，均值居中（范数近似守恒）、'
                        '不硬删任何槽（多器官题不丢上下文），整图第 37 槽中性=1。gamma '
                        '零初始化 ⇒ 初始态与 organ-only **逐位相等**（同 Organ2DResidual/'
                        'OrganEvidence 的家法），训练后 describe()[route_gamma] 直接回答'
                        '「路由学没学起来」。无绑定的题（15.7%%）route_mask 全 False ⇒ 门控'
                        '恒等、退回 organ-only。要器官槽 on（--mode state-only 与 '
                        '--organ-slots off 无槽可缩放）；**不要求 --slices>0**（direct3d '
                        '的 SLICES=0 也成立）；与所有臂正交可组合。对照臂 = 同参数不 '
                        '--organ-route（step 0 两者逐位相等，差异只随 gamma 训练显现）')
    p.add_argument('--organ-route-visual', action='store_true',
                   help='把 --organ-route 的全局标量 gamma 换成**每器官一个视觉条件权重**：'
                        'a_k = bias_k + proj_k·v_k/√D，scale=1+tanh(a_k)·(m−mean m)，'
                        '再对 Δ 做样本内去均值（36 个 scale 均值恒 1）。bias_k 是器官静态'
                        '先验，proj_k·v_k 让同一器官在不同图上拿到不同权重；bias/proj '
                        '零初始化 ⇒ step 0 与不开路由逐位相等。全 bias 相等且 proj=0 时'
                        '严格退化为标量版。隐含 --organ-route；整图第 37 槽中性、无绑定题'
                        '恒等、不要求 --slices>0、与所有臂正交。对照臂 = 同参数 --organ-route'
                        '（标量版）或不开路由')
    p.add_argument('--organ-route-perorgan', action='store_true',
                   help='把 --organ-route 的全局标量 gamma 换成**每器官一个静态标量**：'
                        'g_k = tanh(bias_k)（36 个参数，不看视觉特征），'
                        'scale_k = 1 + g_k·(m_k − mean_j m_j)，再对 Δ 做样本内去均值'
                        '（36 个 scale 均值恒 1）。全 bias 相等时严格退化为标量版；bias '
                        '零初始化 ⇒ step 0 与不开路由逐位相等。隐含 --organ-route；整图第 37 '
                        '槽中性、无绑定题恒等、不要求 --slices>0、与所有臂正交；与 '
                        '--organ-route-visual 互斥（同一门控只能有一种权重参数化）。'
                        '对照臂 = 同参数 --organ-route（标量版）或不 --organ-route')
    p.add_argument('--organ-acquire', action='store_true',
                   help='armEV 证据获取策略头（OrganAcquire）：每器官一个 Bernoulli logit '
                        '+ 整图槽，bias/proj/route_w/whole_bias 零初始化 ⇒ σ=0.5 中性探索。'
                        '动作 = 取哪些**在场**器官证据 + 整图（acquirable，代价 1 单位）；'
                        '未取槽 has_feat=False + organ_box=-1（证据退回 missing）。要配 '
                        '--organ-evidence 与整图槽（n_vis==n_organs+1）；dataset 的 ev 扩成'
                        '「在场器官+整图」全集（与绑定口径相反），route_mask 解耦供获取头'
                        '题绑定偏好。配 --loss evgrpo 使用；既有臂不开此开关逐位不变')
    p.add_argument('--prompt-no-count', action='store_true',
                   help='题面口径（2026-10-31 拍板「一律去掉数字」）：切片句一律改成'
                        '无数量版（「You are given axial CT slices」），与 --slices 解耦 —— '
                        '换切片数不再隐换题面。新臂默认开（run_b.sh 的 PROMPT_NO_COUNT=1）；'
                        '已登记旧臂 eval 与续跑按 run_info/ckpt 记账走旧口径（缺键 = 旧）')
    p.add_argument('--option-perm', type=int, default=0, metavar='SEED',
                   help='choice 题的选项轮转种子（0=关，默认）。非 0 ⇒ 每道题按 '
                        'sha256(seed, qid) 得到一个确定性置换，把选项描述在字母位上轮转'
                        '（字母不动、尾行不动、K 不动，只有「哪个描述坐在哪个字母上」变）。'
                        'noul 与 score 一律不动（score 的档位序是语义，RPS 依赖它）。'
                        '🔴 与 --prompt-no-count 不同：eval 时**命令行说了算**，不被 run_info 覆盖 '
                        '—— 它不是权重的属性，是这一次评估的属性（同一个 ckpt 换个呈现方式再问一遍），'
                        '正是要拿它做「置换不变性闸门」：跑两遍（0 与 1）比 flip 率与 p_gold 漂移。'
                        '置换只依赖 (qid, seed) ⇒ 跨进程/跨解释器稳定，分片 eval 各片自洽。'
                        '🔴 键是 qid 而不是 (study, qid)：同一题名在各卷上必须拿到同一个置换，'
                        '否则 qspace 的跨卷答案空间断言会当场 FATAL'
                        '（见 dataset.option_permutation 的 docstring）。')
    p.add_argument('--vis-tokens', default='all', choices=['all', 'organs', 'whole'],
                   help='哪些特征行进序列（A3/A4）。all（默认）= 主臂原样 36 器官 + 整图；'
                        'organs = A4「去掉整图 token」：projector 建成 36 槽，npz 的第 37 行'
                        '不读也不进序列（顺带取消 organ_evidence 的整图回退支 —— 无绑定题'
                        '退回「只有器官向量、没有证据 token」，计 oev_no_whole_slot）；'
                        'whole = A3「只用整图一个 token」：序列里只有 1 个视觉 token，'
                        'projector 建成 (n_organs=0, has_whole=True)，切片通道/证据/融合/路由'
                        '全部必须关（否则「一个 token」是假的）。收窄必须真是收窄：在没有整图'
                        '行的缓存（pretrain）上用 organs/whole 当场拒 —— 与主臂逐位相同却挂上'
                        '消融标签，比没有消融更坏。eval 以 run_info 为准（它是权重的属性）')
    p.add_argument('--organ-index', default='on', choices=['on', 'off'],
                   help='槽位身份嵌入（A1 的前半）。off = `OrganProjector(index=False)`：'
                        '`nn.Embedding(n_tok, D)` 换成**一个共享向量**加到每个槽上 —— 它被 '
                        'MLP 末层 bias 完全吸收，所以对模型就是「没有任何一列携带器官身份」。'
                        '另一半是 --slot-perm（固定行序），两半都在才是这条臂。36 行向量、'
                        'MLP、missing 占位一律不动 —— 撤的是索引，不是内容。'
                        'eval 以 run_info 为准（参数名与形状都变了：organ_shared vs organ_emb.weight）')
    p.add_argument('--slot-perm', type=int, default=0, metavar='SEED',
                   help='逐 study 槽序置换种子（A1 的后半，0=关）。非 0 ⇒ 每卷按 '
                        'sha256("slot", seed, study_id) 得到 36 个器官行的一个置换，'
                        '特征行 / organ_box / organ_zprof 跟着走，按器官名查行号走逆映射 —— '
                        '于是「第 20 槽是肝」这件事不再成立，而向量本身一个没少。'
                        '键是 study 而不是 qid：特征与几何都是逐卷的，同一卷的所有题必须看到'
                        '同一个槽序。整图行钉在最后一位不动（`_evidence_embeds` 按 '
                        'k==n_organs 认它）。用 sha256 不用内置 hash()：后者带 PYTHONHASHSEED '
                        '盐 ⇒ 分片 eval 各片行序不同、合起来才错。eval 以 run_info 为准')
    p.add_argument('--organ-slots', default='on', choices=['on', 'off'],
                   help='B 支：器官向量（36/37 槽）是否自己进 LLM 序列。on（默认）= 器官槽'
                        '经 projector 进序列（--fuse 是另加的融合通道）；off = ClinFusion '
                        '原样：序列里没有器官槽，3D 只以融合 K/V 身份进入模型（Q=切片 token）。'
                        'off 必须配 --fuse（3D 的唯一入口）与 --slices>0；'
                        '两版对照 = 同参数 on vs off（融合都开着，只看槽的有无）')
    p.add_argument('--train-json', default=None,
                   help='训练数据面：导出 train_val.json（user 题面逐字取自文件，'
                        'assistant 应答段算 answer-only CE）。与 --eval-json 成对给；'
                        '`internal`/`off` = 退回内部题面（keep TSV + out_v3）')
    p.add_argument('--eval-json', default=None,
                   help='评估数据面：导出 test.json（question/answer/type）。'
                        '`internal`/`off` = 退回内部题面。eval 子命令以 run_info 为准')
    p.add_argument('--loss', default='combined',
                   choices=['ce', 'combined', 'racd', 'grpo', 'dpo', 'evgrpo'],
                   help='训练损失：combined（默认）= 按题型合成（choice CE+可接受集质量+'
                        'Brier / noul BCE / score NLL+RPS，见 losses.py），直接对单 token '
                        '候选集打分；ce = 配 --train-json 时走 assistant 应答段的 '
                        'answer-only token CE（Vision-Jev 形式；不配 --train-json 时是'
                        '逐题候选 CE）；racd = SFT **之后**的证据视图校准阶段（combined '
                        '逐视图当 task 项 + 两条视图间 hinge，见 --racd-*）；'
                        'grpo = SFT **之后**的 RL 阶段（RAC arXiv:2605.16999：每题采 G 个 '
                        'rollout、组内归一化优势，校准项走**奖励整形**不走可导 hinge，'
                        '见 --grpo-*）；dpo = 跨视图先验感知硬负例 DPO（不采样：chosen=gold、'
                        'rejected=blind 视图的先验答案，都在 full 视图下取参考 log-ratio，'
                        '见 --dpo-*）；evgrpo = armEV 证据获取 RL（动作 = 取哪些器官证据子集，'
                        '输入侧 rollout 保住 SFT 答案塌方后的多样性；奖励 = correctness − '
                        'λ_cost·|S| − β_cal·Brier，答案头 λ_task·combined 微调 + β_kl 弱锚，'
                        '见 --ev-*）。换损失 = 换一条臂，对照必须同 rules/split/步数')
    p.add_argument('--organ-crop', default='off', choices=['off', 'z', 'bbox', 'mask'],
                   help='B 支定位臂：切片按当前题绑定器官的 RADAR mask 取（分位 0.5/99.5）'
                        '—— z=按 bbox 的 z 范围选层（平面全幅 256×384）；bbox=z 选层 + '
                        '平面裁到器官并集 bbox；mask=按 organ_zprof 的逐层器官体素数选层'
                        '（平面全幅；旧缓存缺该列时逐题回退 z 取层）。读数都是「器官先验'
                        '（+高分辨率）」的合成增益，**不是**模型自主定位；对照必须 '
                        'ORGAN_CROP=off 同参数。要配 --slices>0')
    p.add_argument('--crop-margin', type=float, default=0.1,
                   help='裁剪每边按边长比例外扩（z 每边至少 1 层；平面最小边 32 体素后 clamp 边界）')
    p.add_argument('--crop-cache-n', type=int, default=24,
                   help='裁剪臂切片缓存的请求层数 Mreq=max(--slices, 本值)；缓存的是**全幅**整卷'
                        '切片，裁剪在使用时做 —— 换 margin/换 --slices 都不重刷切片缓存')
    p.add_argument('--organ-map', default=DEFAULT_ORGAN_MAP,
                   help='题名 -> 绑定器官静态表（analysis/organ_crop_map.py 生成）；'
                        'eval 按 run_info 记的 sha256 核对复原，命令行不许偷换绑定表')
    p.add_argument('--cache-root', default=None)
    p.add_argument('--cache-sha12', default=None)
    p.add_argument('--n-organs', type=int, default=N_ORGANS)
    p.add_argument('--only-prefixes', default=None,
                   help='逗号分隔，限定名单 `prefix` 列的取值（出厂题带下划线：finding_/score_/'
                        'side_/organ_；新题是族名：compartment/…）。整串相等比较，'
                        '**不是题名前缀**（--only-prefixes organ_prostate 会筛出 0 道题）')
    p.add_argument('--keep-tsv', default=None)
    p.add_argument('--newq-keep', default=None)
    p.add_argument('--no-check-cache', action='store_true')
    p.add_argument('--tok', default=None)
    p.add_argument('--tiny', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='train: 最多 optimizer step；eval: 最多例数；0=不限')


def _fill_defaults(a):
    from .dataset import DEFAULT_CACHE, DEFAULT_KEEP_TSV, DEFAULT_NEWQ_KEEP
    a.cache_root = a.cache_root or DEFAULT_CACHE
    a.keep_tsv = a.keep_tsv or DEFAULT_KEEP_TSV
    a.newq_keep = a.newq_keep or DEFAULT_NEWQ_KEEP
    for attr in ('train_json', 'eval_json'):
        v = getattr(a, attr, None)
        if isinstance(v, str):
            v = v.strip()
            setattr(a, attr, None if v.lower() in ('', 'internal', 'off', 'none') else
                    os.path.abspath(os.path.expanduser(v)))
        else:
            setattr(a, attr, None)
    return a


def main(rest):
    ap = argparse.ArgumentParser(prog='radev train', add_help=True)
    sub = ap.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='训练（bf16 + 梯度累积 + 续跑 + --limit 干跑）')
    _add_common(pt)
    pt.add_argument('--out', required=True)
    pt.add_argument('--llm', default=None,
                    help='模型本地目录（A/B 两支同一 Qwen3.5-*，A 支不喂切片）')
    pt.add_argument('--dtype', default='bfloat16')
    pt.add_argument('--freeze-llm', action='store_true')
    pt.add_argument('--attn', default=None, help='attn_implementation，如 sdpa/flash_attention_2')
    pt.add_argument('--bs', type=int, default=8)
    pt.add_argument('--accum', type=int, default=1)
    pt.add_argument('--grad-ckpt', action=argparse.BooleanOptionalAction, default=True)
    pt.add_argument('--lr', type=float, default=2e-5)
    pt.add_argument('--lr-sched', default='const', choices=['const', 'cosine'],
                   help='LR 调度：const（默认，所有既有臂）= 全程常数；cosine = '
                        'CosineAnnealingLR 从 --lr 余弦衰减到 0（RAC §3.1 的口径就是 '
                        '1e-6 + cosine）。cosine 必须有衰减终点（--lr-total，缺省取 '
                        '--limit）。调度状态存进 ckpt，续跑逐位接得上；const 与 cosine '
                        '之间不许互相 --resume（lr 曲线接不上而前向不报错）')
    pt.add_argument('--lr-total', type=int, default=0, metavar='STEPS',
                   help='cosine 的 T_max（总步数）。0 = 取 --limit；两者都没有当场拒 —— '
                        '没有终点的 cosine 跑过头之后会按周期把 lr 升回去，那是另一条臂')
    pt.add_argument('--wd', type=float, default=0.0)
    pt.add_argument('--clip', type=float, default=1.0)
    pt.add_argument('--epochs', type=int, default=1)
    pt.add_argument('--seed', type=int, default=0)
    pt.add_argument('--log-every', type=int, default=50)
    pt.add_argument('--save-every', type=int, default=0)
    pt.add_argument('--resume', default='auto', help='auto|off|<ckpt 路径>（同一次跑的延续，含优化器）')
    pt.add_argument('--init-from', default=None, metavar='CKPT',
                   help='从**另一段**训练的 ckpt 只取权重（典型：projector-only 对齐段 -> 解冻 LLM 全参段）。'
                        '优化器与步数全新；LLM/编码器/训练集与来源不一致当场拒，核对结果写 init_from.json')
    pt.add_argument('--racd-views', type=int, default=3, metavar='M',
                   help='每题的证据视图数 M（槽 0 恒 full；槽 1 = 屏蔽绑定器官；'
                        '槽 2+ = 各屏蔽一个不绑定的在场器官）。抽不出的槽退回 full，'
                        '比较项按 kind 跳过。显存 = M× 单题前向，BS 要相应下调')
    pt.add_argument('--racd-mask-whole', action='store_true',
                   help='target_mask 视图里把**整图全局槽**（第 n_organs+1 个 RADAR 槽）'
                        '也一起屏蔽。默认关 = 老口径（整图槽豁免），既有臂逐位不变。'
                        '关着的代价是实测的：racd1 的 gate_frac 只有 0.0431(step20)→'
                        '0.0194(step2000)，λ_corr=0.3 被稀释成 ≈0.012、corr 轴基本空转 —— '
                        '因为该臂切片 token 不进序列，遮掉绑定器官槽之后 whole_feat 仍在，'
                        '病灶从全局摘要里照样读得出来，反事实留了一条泄漏通路。'
                        '整图槽只进 target_mask，绝不单独当 distractor（全局摘要对任何题'
                        '都不是「无关证据」，对它施加不变性约束是错的约束）。'
                        '序列里没有第 37 槽时（whole_feat 没进缓存 / direct3d 的 n_vis=0）'
                        '当场 SystemExit，不静默降级')
    from .losses import RACD_TASK_VIEWS
    pt.add_argument('--racd-task-views', choices=('all', 'no_target', 'non_gated'),
                    default=RACD_TASK_VIEWS,
                    help='**答案监督（combined_loss）给到哪些证据视图** —— 这一项决定'
                         'racd 段到底是「反事实」还是「增广」，不是校准 λ 那种量级旋钮。'
                         'all = 历史口径（已跑的 out_racd/2B-oev-s32-* 两条臂，逐位可复现）：'
                         '每个视图都吃 CE，于是 target_mask 行上同一份 logits 被两头拉 —— '
                         'CE 要 c→1、rank 要 c ≤ c_full − m_rank，符号相反而权重比约 '
                         '1 : 0.0005，rank 实际被 CE 压住。'
                         'no_target = RAC（arXiv:2605.16999）/RLCD（arXiv:2609.38850）原式：'
                         '屏蔽视图只进比较项，不进答案损失（矛盾整条视图消掉）。'
                         'non_gated = 折中：遮蔽后答案没变坏（q_full == q_tgt，门控为假）的组'
                         '继续吃 CE（那是合法增广），变坏了的组退出 CE 让给 rank 项'
                         '（矛盾逐组消掉）。distractor 视图三种口径下都吃 CE —— 屏蔽无关器官'
                         '本来就该答得一样，两项同向无矛盾。'
                         f'槽 0（full）任何口径下都不摘，batch 不会被清空。诊断列 '
                         'task_keep = 吃答案监督的行占比（all 恒 1.0）')
    pt.add_argument('--racd-lam-rank', type=float, default=0.2,
                   help='质量条件式排序 hinge 的权重（RAC 的 λ_rank 同量级）')
    pt.add_argument('--racd-lam-null', type=float, default=0.1,
                   help='干扰器官不变性 hinge 的权重（防「见 mask 就降置信度」的捷径）')
    pt.add_argument('--racd-m-rank', type=float, default=0.05,
                   help='排序 hinge 的间隔 m_rank：要求 c_full − c_tgt ≥ 本值（RAC 同值）')
    pt.add_argument('--racd-m-null', type=float, default=0.03,
                   help='不变性 hinge 的容差 δ：c_full − c_dist 超过本值才罚（RAC 的 m_corr 同量级）')
    from .losses import RACD_BLIND, RACD_M_BLIND
    pt.add_argument('--racd-blind', action='store_true',
                   help='给每题多加一个**盲视图**槽（恒占最后一槽）：把全部在场视觉槽（含整图'
                        '槽）一次屏掉。要 --racd-views>=3，且这条臂序列里有第 37 整图槽时要'
                        '--racd-mask-whole 一起开（否则 whole_feat 还在，「盲」是假的 ⇒ 量到的'
                        '是泄漏不是语言先验，views.py 当场炸）。盲视图无条件退出答案监督，'
                        '所以 VIEWS=4+blind 吃 CE 的那三行与 VIEWS=3 逐位同构')
    pt.add_argument('--racd-lam-blind', type=float, default=RACD_BLIND,
                   help='盲视图信心下降 hinge 的权重：要求 c_full − c_blind ≥ m_blind。'
                        '>0 而没开 --racd-blind 是废臂（批次里没有 blind 槽 ⇒ 项恒 0），'
                        '_check_racd 在配置期就拦')
    pt.add_argument('--racd-m-blind', type=float, default=RACD_M_BLIND,
                   help='盲视图 hinge 的间隔：盲掉之后信心至少掉这么多才算「这份信心是证据'
                        '支撑的」。该设多大取决于 d_blind 的分布，而它从没落过盘 ⇒ 开了 blind'
                        '就无条件记六点 CDF（bgap_lt_*）；权威口径是 CDF 不是 viol_blind，'
                        '但两者共用 n_blind_gate 权重 ⇒ 逐位相等，是一条可用的 step-20 自校验')
    pt.add_argument('--racd-tau', type=float, default=0.5,
                   help='器官组温度分层指数：p_g ∝ n_g^tau（1=全局均匀，0=各组等量）。'
                        '均匀采样下最小组（Uterus and ovaries，78 行）在 2000 步里只摊到 '
                        '~22 次 target_mask，那一组的 Δ_target 测不出来')
    pt.add_argument('--racd-min-frac', type=float, default=0.005,
                   help='每组配额地板（占全量行数比例）。与 tau 取大者，再被 --racd-max-repeat 封顶')
    pt.add_argument('--racd-max-repeat', type=int, default=8,
                   help='一个组每 epoch 最多被复读几遍（配额上限 = n_g × 本值）。'
                        '抬稀小组靠的是复读，封顶是防 78 行被训上百遍')
    from .losses import RACD_CAL, RACD_CAL_MODE, RACD_CAL_PI_HI, RACD_CAL_PI_LO, RACD_CAL_PI_W
    pt.add_argument('--racd-lam-cal', type=float, default=RACD_CAL,
                   help='可靠性项权重：lam_cal · mean_rows v(π_q)·(p_max − y)²，'
                        'y = 1[argmax == gold]（训练期 gold 已知，不碰 test）。'
                        '与 CE 反号 ⇒ 必须配 --racd-cal-warm 两阶段点火。'
                        'armG/armH 用 0.5（与 CHOICE_CE=0.5 同量级）')
    pt.add_argument('--racd-cal-mode', choices=('flat', 'pi'), default=RACD_CAL_MODE,
                   help='flat = 全卷等权（armG，方案 B）；pi = 按训练侧先验 π_q 分带加权'
                        '（armH，方案 F：中先验带 [0.55,0.85] 火力全开，带外降权）。'
                        '两者的 λ_cal 必须同值，否则 G/H 差两个因子')
    pt.add_argument('--racd-cal-warm', type=int, default=0, metavar='STEPS',
                   help='两阶段点火步数：前 warm 步 λ_cal 恒 0（纯 armF 口径），随后再 warm 步'
                        '线性升到 --racd-lam-cal。0 = 第 1 步就全开（会把 CE/cal 对抗写进早期'
                        '权重，只在扫这个风险时用）。700 步臂用 350')
    pt.add_argument('--racd-cal-pi-lo', type=float, default=RACD_CAL_PI_LO,
                   help='π 加权带的下界（含）。定标来自 out_logs/prior_per_qid.tsv 的分桶表：'
                        '|偏差|>0.02 的单元全落在 π∈[0.48,0.72]')
    pt.add_argument('--racd-cal-pi-hi', type=float, default=RACD_CAL_PI_HI,
                   help='π 加权带的上界（含）。π≥0.8 的桶 armD 已做到 |偏差|≤0.008，再加压'
                        '会把符号翻回去（SFT 在同格是欠自信那一侧）')
    pt.add_argument('--racd-cal-pi-w', type=float, default=RACD_CAL_PI_W,
                   help='带外行的权重。**不许设 0**：摘掉带外行等于同时改了 batch 构成，'
                        'armH 与 armF 就不止差一项了。校验在 _check_racd')
    pt.add_argument('--racd-prior-table', default=None, metavar='TSV',
                   help='逐题训练侧先验 π_q 表（qid<TAB>pi_mode，只由 train+val 标签生成 ⇒ '
                        '不是 oracle-on-test）。产物 out_logs/prior_per_qid.tsv，重建：'
                        'python3 out_logs/pi_strata.py --regen。cal_mode=pi 时必给，'
                        '且必须覆盖训练集全部题 —— 缺题当场停，不静默补 0')
    from .losses import (GRPO_ALPHA, GRPO_BETA, GRPO_CORR, GRPO_LAM_TASK, GRPO_M_CORR,
                        GRPO_M_RANK, GRPO_RANK, GRPO_SAMPLES, GRPO_W_TASK)
    pt.add_argument('--grpo-samples', type=int, default=GRPO_SAMPLES, metavar='G',
                   help='每题采几个 rollout（RAC 的 n，原文 8）。**不花额外前向**：'
                        'score_exact 一次前向就给出全部 K≤11 个候选的精确 π，G 只是从'
                        '这 K 个里 multinomial 抽几个。G<2 当场拒（组内归一化没有 σ）')
    pt.add_argument('--grpo-lam-task', type=float, default=GRPO_LAM_TASK,
                   help='task 项（combined proper score）在总损失里的权重。默认 0 = 纯 '
                        'GRPO（RAC Eq.12 没有监督项，且 SFT 段已结束）；设 >0 就是另一条臂。'
                        '无论取值，task 都照算照记进日志 —— 不看影像的地板漂到哪去了'
                        '得能查到。与 racd 的 λ_rank=0 同一条纪律')
    pt.add_argument('--grpo-w-task', type=float,
                   default=float(os.environ.get('GRPO_W_TASK', GRPO_W_TASK)),
                   help='r^task 在 r̃ 里的权重（RAC Eq.11 原文写死 1，没有这个旋钮）。'
                        '_reward 是**采到的答案**的函数 ⇒ E_p[r]=Σ_k p_k·r(k) 对 p 线性 ⇒ '
                        '单纯形上顶点最优 ⇒ 这一项是纯锐化压力，方向与提校准相反，而 '
                        'λ_task/β/λ_corr 只能改「多快走到顶点」。设 0 把锐化项从优势里整个'
                        '摘掉，pg 只携带 r^corr 的反事实信号；准确率那时由 λ_task 的 '
                        'proper score + β·KL 锚住。_reward 照常算（gate 的 r0>r1 读它），'
                        'r_task 读数照记原值 ⇒ 跨臂可比。默认 1.0 = 既有六条臂逐位不变')
    pt.add_argument('--grpo-lam-rank', type=float, default=GRPO_RANK,
                   help='RAC Eq.4–6 的组内排序整形权重 λ_rank（原文 0.2；消融 0/0.1/0.2/'
                        '0.3/0.5 里 0.2 最优，acc 0.656 / ECE 0.340）。这是 RAC **提升 '
                        'acc** 的那一项 —— racd 臂没有它（racd 不采样，表达不了「同题不同 '
                        'rollout」这条轴）')
    pt.add_argument('--grpo-m-rank', type=float, default=GRPO_M_RANK,
                   help='排序整形的间隔 m_rank（RAC 同值 0.05）：答对者的置信度要比答错者'
                        '高出本值才不罚')
    pt.add_argument('--grpo-lam-corr', type=float, default=GRPO_CORR,
                   help='RAC Eq.7–9 的干净/退化支路整形权重 λ_corr（原文 0.3；消融 0→0.3 '
                        '把 ECE 0.340→0.284，**0.4 让 acc 从 0.656 塌到 0.589**）。>0 时'
                        '要求多视图批次（--racd-views>=2）；纯 rank 臂请设 0，能省掉 M 倍前向')
    pt.add_argument('--grpo-m-corr', type=float, default=GRPO_M_CORR,
                   help='退化支路的间隔 m_corr。**论文没报这个数**（m_corr/α/β_KL/ε 全部'
                        '未给），0.05 是本仓库自己定的，只能自证不能引用')
    pt.add_argument('--grpo-alpha', type=float, default=GRPO_ALPHA,
                   help='severity 斜率 α（RAC 用连续 s，我们的退化是二值的 ⇒ 默认 0）。'
                        '负值当场拒：符号约定是「越退化越该不自信」')
    pt.add_argument('--grpo-beta', type=float, default=GRPO_BETA,
                   help='RAC Eq.2 的 KL 锚系数 β_KL。**论文没报这个数**，0.04 是本仓库'
                        '自己定的。>0 要参考策略：给 --grpo-ref（续跑时必须给，'
                        '--init-from 会被丢掉）。设 0 就不 deepcopy、不跑那次参考前向')
    pt.add_argument('--grpo-ref', default=None, metavar='CKPT',
                   help='KL 锚的参考策略权重（典型 = SFT 段的 ckpt_last.pt）。锚必须是'
                        '**SFT 起点**：续跑时用 --init-from 会被 RL 进度覆盖，KL 就退化成'
                        '「别离开 500 步前的自己」，那不是锚')
    pt.add_argument('--grpo-balance', default='group', choices=['off', 'group'],
                   help='off = StudyShuffledSampler（与 SFT 臂同采样）；group（默认）= '
                        'GroupStratifiedSampler 器官组分层。**代价**：分层器读的是 '
                        'MultiViewDataset 的 group_of，所以 group 会连带把每题行数乘上 M '
                        '（--racd-views），不是免费的。稀有组的曝光靠它，不靠奖励加权 —— '
                        '组内归一化本身就是平衡机制')
    pt.add_argument('--grpo-kind-alpha', type=float, default=1.0, metavar='A',
                   help='题型平权采样的指数：每个题型本 epoch 的目标行数 T_k ∝ n_k^A。'
                        'A=1（默认）= 按行数，与既有臂逐位一致；A=0 = 三个题型各拿 1/3 '
                        '的行。报数按题型平权（noul mBAcc / choice mF1 / score MAE-QWK '
                        '各一列），而训练行数是 66.7/28.5/4.7 —— A<1 就是把训练曝光对齐'
                        '到报数口径。**代价**：A=0 时 score 那 2,690 行每 epoch 被过 ~7 遍'
                        '（上限 max_repeat=8），会被背下来。要配 --grpo-balance group'
                        '（题型平权叠在绑定器官分层之上，不牺牲小组曝光）')
    pt.add_argument('--grpo-no-gate-corr', action='store_true',
                   help='关掉 r^corr 的质量闸 1[a_clean=对, a_degraded≠对]（RAC/RLCD 的'
                        'gate）。关掉 = 每一对都开火，包括「干净支路本来就答错」的那些 —— '
                        '那是在要求模型对一个它没答对的题降置信度。默认开')
    pt.add_argument('--grpo-corr-cand', default='cross', choices=('cross', 'same'),
                   help='r^corr 那个 hinge 比的是谁的 π。cross（默认 = 已跑过的六条 RL 臂'
                        '口径，逐位可复现）：π_full(干净端自己采的候选) vs π_mask(退化端自己'
                        '采的候选) —— 两个**不同答案**的概率相减。same：两端都改在干净端那次'
                        '抽样的候选上取 π，与 racd_loss 的 c_f/c_t 同一个对法。cross 有结构'
                        '性死角：gate 要求「干净端答对 ∧ 退化端答错」，hinge 要求「退化端给'
                        '那个错答案的信心 ≥ 干净端给对答案的信心 − m_corr」，后者在干净端'
                        'π(gold)→1 时恒不成立 ⇒ 训练越成功这一项越熄火（在跑的臂实测 '
                        'r_corr=−0.0099，只占 r_std=0.1605 的 6%）')
    pt.add_argument('--grpo-task-views', default=None,
                    choices=('all', 'no_target', 'non_gated'),
                    help='答案监督（λ_task 那一项）给到哪些视图。默认 None = 历史口径，'
                         '所有视图都吃 CE，已跑的六条臂逐位不变。选 same 时这是个真冲突：'
                         'gate 为真 ⇒ 干净端采到的就是 gold ⇒ hinge 压的是 π_mask(gold)，'
                         '而 CE 在同一个 logit 上要 π_mask(gold)→1，两项符号相反、互相抵消。'
                         'no_target = RAC/RLCD 原式（反事实视图只进比较项）；non_gated = '
                         '只摘组内**任一次 rollout 开了闸**的那些行（判据 gate.any(dim=1)，'
                         '因为这里的 gate 逐 (组,样本) 算，一次开火冲突就已落在那份 logits 上）。'
                         '只在 --grpo-lam-corr>0 且多视图时可用，否则直接报错')
    pt.add_argument('--grpo-view-weight', default='uniform',
                    choices=('uniform', 'grounded'),
                    help='pg 的视图权重。uniform（默认）= 历史口径，所有视图均匀 mean，'
                         '既有臂逐位不变。grounded = 「多视图的选择由奖励实现」：每个视图的 '
                         'pg 权重 = relu(该视图平均 r^task − blind 平均 r^task)，blind 槽恒 0 '
                         '且退出答案监督。要 --racd-blind + --racd-views>=3（批次里有 blind 槽）。'
                         'armFR 用它：在 armF 之上 KL 锚定、只留 r^task、奖励决定视图权重')
    from .losses import DPO_BETA, DPO_LAM_CE
    pt.add_argument('--dpo-beta', type=float, default=DPO_BETA,
                   help='DPO 参考锚强度 β（losses.dpo_loss）。loss=−log σ(β·[chosen 与 rejected '
                        '的参考 log-ratio 差])。越大越贴近参考策略、越不锐化，兼作校准护栏。'
                        '>0 要参考策略：--dpo-ref（续跑）或 --init-from（首跑，deepcopy 即锚）')
    pt.add_argument('--dpo-lam-ce', type=float, default=DPO_LAM_CE,
                   help='在 full 视图叠一份 combined proper score 作监督锚的权重。默认 0 = '
                        '纯 DPO；blind 视图无条件排除（要求无证据读出答对 = 教先验）')
    pt.add_argument('--dpo-ref', default=None, metavar='CKPT',
                   help='DPO 参考策略权重（典型 = SFT 段 ckpt_last.pt）。锚必须是 SFT 起点：'
                        '续跑时用 --init-from 会被 RL 进度覆盖，log-ratio 就退化成「别离开 '
                        '500 步前的自己」，那不是锚')
    from .losses import (EV_SAMPLES, EV_LAM_COST, EV_BETA_CAL, EV_BETA_KL, EV_LAM_TASK)
    pt.add_argument('--ev-samples', type=int, default=EV_SAMPLES,
                   help='每题采 G 个证据子集（输入侧 rollout）。G>=2 才有组内 σ；动作改的是'
                        '输入（取哪些器官证据），不是答案候选 —— 单 token SFT 答案塌方后'
                        '仍有组间多样性。代价：每步 G 次答案前向 + β_kl>0 一次参考前向')
    pt.add_argument('--ev-lam-cost', type=float, default=EV_LAM_COST,
                   help='奖励里每取一个器官/整图的代价 λ_cost（r = correctness − λ_cost·|S| '
                        '− β_cal·Brier）。→0 = 语言先验塌方（全取），→n_present = 代价太弱。'
                        '默认 0.02 本仓库自定')
    pt.add_argument('--ev-beta-cal', type=float, default=EV_BETA_CAL,
                   help='奖励里的 Brier 校准罚权重 β_cal。塑造获取策略「取能降低 Brier 的'
                        '证据」；与 combined_loss 里的 Brier 是两个角色（后者训答案头）。'
                        '默认 0.1 本仓库自定')
    pt.add_argument('--ev-beta-kl', type=float, default=EV_BETA_KL,
                   help='答案头 KL 锚强度 β_kl（forward KL，ref 在同一 masked 批上打分）。'
                        '弱于 FR 的 0.04：ref=SFT 没见过 masked 输入，锚太强会压死获取探索。'
                        '>0 要 --ev-ref（续跑）或 --init-from（首跑，deepcopy 即锚）。设 0 '
                        '就不建参考模型、不跑那次前向。默认 0.01 本仓库自定')
    pt.add_argument('--ev-lam-task', type=float, default=EV_LAM_TASK,
                   help='答案监督权重 λ_task（combined_loss 逐题型 proper score，含校准）。'
                        '答案头与获取头一起微调；设 0 = 纯获取 PG（答案头只靠 KL 锚）。'
                        '默认 1.0')
    pt.add_argument('--ev-ref', default=None, metavar='CKPT',
                   help='armEV KL 锚的参考策略权重（典型 = SFT 段 ckpt_last.pt）。锚必须是'
                        'SFT 起点：续跑时用 --init-from 会被 RL 进度覆盖，KL 就退化成「别'
                        '离开 500 步前的自己」，那不是锚')
    pt.set_defaults(fn=cmd_train)

    pe = sub.add_parser('eval', help='评估（score_exact 精确打分 + 指标 + 地板 McNemar）')
    _add_common(pe, rules_required=False)
    pe.add_argument('--run', required=True, help='训练产物目录（读 run_info.json + ckpt_last.pt）')
    pe.add_argument('--out', default=None)
    pe.add_argument('--llm', default=None)
    pe.add_argument('--dtype', default='bfloat16')
    pe.add_argument('--freeze-llm', action='store_true')
    pe.add_argument('--attn', default=None)
    pe.add_argument('--bs', type=int, default=8)
    pe.add_argument('--score-oneshot', type=int, choices=(0, 1), default=1, metavar='0|1',
                    help='1 = 本批候选全为单 token 时每题只推一次前缀，K 个分数取自同一行 logits'
                         '（symbol_mode 出口即保证如此；整卷 meanK=3.061 ⇒ 少跑 67% 的前缀）；'
                         '0 = 逐候选复制前缀的老口径，留给新旧读数对拍。'
                         '候选里有任何多 token（旧导出 `Answer: …` 族）时两种取值都走老路，'
                         '所以这个开关只在 sym 卷上有实际效果')
    pe.add_argument('--num-shards', type=int, default=1, metavar='N',
                    help='把 split 切成 N 个连续行块，本进程只跑 --shard 那一块，'
                         '落 verdicts_shard*/eval_shard* 两份产物后**直接退出**（不出 report，'
                         'report 由 --merge-shards 那一步统一出）。默认 1 = 现有单卡口径，逐字不变')
    pe.add_argument('--shard', type=int, default=0, metavar='I',
                    help='本进程跑第几块（0..N-1）')
    pe.add_argument('--merge-shards', action='store_true',
                    help='读 --out 里全部 eval_shard*_of_*.json 并成 (pred,p_pos,p_dist) 后走'
                         '原来的 report 尾段。缺片/块不连续/有重叠当场 SystemExit —— '
                         '按缺片的分母出报告等于换了个更小的 test 集，而报告里看不出来。'
                         '不推理，但仍会载入 ckpt（复用 run_info 复原臂参数那条路径）')
    pe.add_argument('--eval-keep-bound', choices=('off', 'bound', 'bound_whole'),
                    default='off',
                    help='评估期只留本题绑定器官槽（views.KeepBoundDataset），量「视觉信息'
                         '只走绑定器官那条路够不够答题」。与 RACD 的 target_mask 同一套'
                         '注入点但方向相反：RACD 屏**绑定**器官造反事实，这里屏**非**绑定'
                         '器官。🔴 不省 token：屏蔽只是把该槽换成可学 missing 向量，序列'
                         '长度一位不变 —— 这是信息充分性读数，不是效率读数。'
                         'bound = 连整图第 37 槽一起屏（**可解释口径**）；bound_whole = '
                         '留整图槽，但 whole_feat 是全卷摘要、会漏答案（压垮 racd1 '
                         'gate_frac 的同一条通路），它下面读到「没掉」分不清原因，只当对照。'
                         '无绑定的题（NO_GROUP，实测 15.7%）与绑定器官全不在场的题一个槽'
                         '都不屏，各自单独计数不混进读数。off（默认）= 不套包装，逐位不变。'
                         '要 RADAR 槽进序列（n_vis>0），direct3d/route 臂当场 SystemExit')
    pe.set_defaults(fn=cmd_eval)

    pf = sub.add_parser('floors', help='只算 ①② 两级地板（零训练）')
    _add_common(pf)
    pf.add_argument('--out', default='.')
    pf.set_defaults(fn=cmd_floors)

    a = ap.parse_args(rest)
    _fill_defaults(a)
    if getattr(a, 'organ_route_visual', False) or getattr(a, 'organ_route_perorgan', False):
        a.organ_route = True
    return a.fn(a)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
