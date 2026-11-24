"""训练题集 -> 序列样本（一序列一题），外加 RADAR 器官嵌入的读取层。

计划出处（`.qoder/plans/earnest-mesa-source.md` 步骤 5、§3.2）：

* **一序列一题**。批次里同一 study 的题各自独立成序列，模型答第 k 题时看不见第 k+1 题的
  ground truth —— 打包会让 `gallbladder: surgically absent ⇒ gallstones: No` 这类
  共现捷径不经影像就能走通。
* **study 级 shuffle**（`StudyShuffledSampler`）：同一 study 的题在批次里连续出现，
  嵌入与记录只读一次；不同 study 之间按 epoch 洗牌。题级 shuffle 会让每条样本都换 study。
* 题集来自两份**逐题名单**，判据都带 `gate` 列（`analysis/gate.py` 的 ADOPTED）：
    `out_inventory/question_keep.tsv`  出厂 733,452 道题（`analysis/question_keep_list.py`）
    `out_newq/newq_keep.tsv`           五族新题（C.26，与出厂链分离，记录在 `out_newq/questions/`）
  两份名单里的 gate 与当前档位不符 ⇒ FATAL（换档必须重跑名单，旧档名单不能冒充新档）。
* `--rules` 是**训练期收窄旋钮**，作用在 keep TSV 的 `rule_class` 列（只有 `finding_*` 非空）
  上，**没有默认值**：出厂白名单是数据决策（plan §8.1），代码不许替用户选。

器官嵌入的来源有三条（`feats`）：
* `cache` —— 读 stage 1 缓存（`radev/train/cache.py`）：`<root>/<sha12>/<split>/<sid>.npz`。
  缓存不存在时给出指向 `run_radar_cache.sh` 的错误，而不是让 25k 次 `__getitem__` 各自报
  FileNotFoundError。
* `live` —— 冻结塔在训练进程里现算（`visual_frontend.py`，plan C.44）：本模块**不自己建塔**，
  `frontend` 由 `loop.py` 按 `--encoder` 构造，一个进程一份；行数是身份的属性（36 器官，
  `radar_ft` 再多一个整图 token）。`mode=state-only`（`n_vis=0`）时允许 `frontend=None`。
* `zeros` —— **冒烟模式**（零嵌入 + `has_feat` 全 False）：只用于验证流水线，
  run info 里必须写明，不许拿它报数。
"""
from __future__ import annotations

import collections
import hashlib
import itertools
import json
import os
import random
import re
import sys

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .. import config
from . import pack
from . import radar_encoder as RE
from . import visual_frontend as VF
from .slice_frontend import union_crop

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_KEEP_TSV = os.path.join(REPO, 'out_inventory', 'question_keep.tsv')
DEFAULT_NEWQ_KEEP = os.path.join(REPO, 'out_newq', 'newq_keep.tsv')
DEFAULT_NEWQ_DIR = os.path.join(REPO, 'out_newq', 'questions')
DEFAULT_CACHE = os.path.join(REPO, 'out_radar_emb')

DEFAULT_TRAIN_JSON = os.path.join(REPO, 'out_dataset_prose_llm', 'train_val.json')
DEFAULT_EVAL_JSON = os.path.join(REPO, 'out_dataset_prose_llm', 'test.json')

_ANSWER_RE = re.compile(r'^Answer:\s*(.+?)\s*$')


def _strip_image_placeholder(user):
    """导出 user 内容 -> 视觉槽之后的题面。`<image>\\n` 是卷级占位符，不进文本 token。"""
    if '<image>' not in user:
        return user
    if user.count('<image>') != 1 or not user.startswith('<image>'):
        raise ValueError(f'<image> 的位置不是唯一的开头：{user[:80]!r}')
    text = user[len('<image>'):]
    return text[1:] if text.startswith('\n') else text


_SLICES_PHRASE_RE = re.compile(r'You are given \d+ axial CT slices')
EXPORT_PROMPT_SLICES = 32


def rewrite_prompt_slices(prompt, slices, no_count=False):
    if not no_count and (slices is None or int(slices) == EXPORT_PROMPT_SLICES):
        return prompt
    new, n = _SLICES_PHRASE_RE.subn('You are given axial CT slices', prompt, count=1)
    if n != 1:
        raise SystemExit(f'要去掉切片句里的数量，但切片句不在：{prompt[:120]!r}'
                         f'（slices={slices} no_count={no_count}）')
    return new


def _cut_at_instruction(block):
    """截到 `Reply with exactly …` 那一行之前；兼容旧的 `Answer: …` 指令与新符号指令。"""
    lines = []
    for ln in block.splitlines():
        if ln.startswith('Reply with exactly'):
            break
        lines.append(ln)
    return lines


def _parse_export_prompt(text):
    """冻结渲染器的三类题面 -> (kind, instructions, options/levels)。

    `options` 是 choice 的 `[(key, description)]`；`levels` 是 score 的完整档位短语。
    同时认两种写法：老语义格式（`- key: desc` / `- level`）与 Jev 单 token 格式
    （`A: desc` / `1: level`）。换模板必须同时改这里（否则解析不出来会比静默错配更早暴露）。
    """
    m = re.search(r'Clinical question: (.*?)\n', text)
    if not m:
        raise ValueError(f'题面里没有 Clinical question 行：{text[:120]!r}')
    instr = m.group(1)
    if 'Severity levels:\n' in text:
        block = text.split('Severity levels:\n', 1)[1]
        levels = []
        for ln in _cut_at_instruction(block):
            if not ln.strip():
                continue
            if ln.startswith('- '):
                levels.append(ln[2:])
            else:
                m2 = re.match(r'^(\d+): (.*)$', ln)
                if not m2:
                    raise ValueError(f'score 档位行不认识：{ln!r}')
                if int(m2.group(1)) != len(levels) + 1:
                    raise ValueError(f'score 档位编号不连续：{ln!r}')
                levels.append(m2.group(2))
        if not levels:
            raise ValueError(f'score 题面没有档位：{text[:120]!r}')
        return 'score', instr, levels
    if 'Options:\n' in text:
        block = text.split('Options:\n', 1)[1]
        opts = []
        for ln in _cut_at_instruction(block):
            if not ln.strip():
                continue
            if ln.startswith('- '):
                key, sep, desc = ln[2:].partition(': ')
            else:
                key, sep, desc = ln.partition(': ')
                if not re.fullmatch(r'[A-Za-z]', key):
                    raise ValueError(f'choice 选项行不认识：{ln!r}')
            if not sep:
                raise ValueError(f'choice 选项行没有 ": "：{ln!r}')
            opts.append((key, desc))
        if not opts:
            raise ValueError(f'choice 题面没有选项：{text[:120]!r}')
        return 'choice', instr, opts
    if ('"Answer: Yes" or "Answer: No"' not in text
            and 'Reply with exactly one letter: a = No, b = Yes.' not in text):
        raise ValueError(f'认不出题型（既非 score/choice，也没有 a/b 尾巴）：{text[:120]!r}')
    return 'noul', instr, None


def option_permutation(qid, seed, k):
    """`(题名, seed) -> range(k) 的一个置换`，跨进程稳定。

    🔴 键是 **qid，不是 `rec['id']`（= `study::qid`）**，这条是实测出来的硬约束，不是审美：
    `_init_json` 的 `qspace` 只按 qid 建表，并要求同一题名在各卷上的 `AnswerSpace`
    逐字段相同 —— 其中包含 `source_texts`（**按字母序**排的描述元组）。按 (study, qid)
    取置换 ⇒ 同一题在不同卷上轮转不同 ⇒ `source_texts` 不同 ⇒ 装载直接 FATAL
    （实测 2026-11-11：`organ_lower 在不同卷上的答案空间不一致`）。
    按 qid 取置换则各卷逐字相同，那条断言原样成立，`self.spaces` / `self.encs`
    这两个按 (kind, qid) 缓存的表也依旧正确 —— 不需要为了这个旋钮放宽任何既有安全网。
    代价是「同一道题在所有病人上用同一个选项序」，而这与现状同级：不轮转时
    gold 字母本来就是 qid 的固定函数，所以按 qid 轮转**不新增**任何记忆面。
    跨 seed 仍然给出同一题的不同置换 ⇒ `ℓ_π`（同题两种呈现的一致性正则）照样能做。

    🔴 用 sha256 而不是内置 `hash()`：后者带 per-process 随机盐（PYTHONHASHSEED），
    而分片 eval 是多进程 ⇒ 同一道题在不同片上会拿到不同置换，那份数据直接废掉
    （且废得很安静：每片自己都是自洽的，只有合起来才错）。
    置换只依赖 (qid, seed)，与批次、分片、epoch 的顺序都无关 ⇒ **分片数不变性成立**，
    这一点与任务 #201 那个「片数会改数」的教训是两回事（那个改的是数值归约顺序）。
    """
    h = hashlib.sha256(f'{seed}:{qid}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(h[:8], 'big'))
    p = list(range(k))
    rng.shuffle(p)
    return tuple(p)


def apply_option_permutation(rec, seed):
    """把 choice 题的**选项描述在字母位上轮转**，返回 `(新 rec, perm)`；不改就 `perm=None`。

    设计是「**字母不动，描述转**」，不是「重排后重新标号」，于是三件事自动成立：
      ① 尾行 `Reply with exactly one option label (a, b, …)` 只依赖 K ⇒ **逐字不变**；
      ② 题面其余部分（切片句 / Patient / Clinical question）逐字不变；
      ③ `cand_ids`、单 token 性质、K 都不变 ⇒ 只换「哪个描述坐在哪个字母上」。
    gold 用**下标**跟着自己的描述走到新字母位，不按文本匹配 —— 实测 16,210 道 choice
    题里 0 条有重复描述，但不拿这个当前提。

    `seed == 0` / 非 symbol 模式 / 非 choice / K < 2 / 置换恰为恒等 ⇒ 原样返回。
    恒等那条很要紧：它保证 `--option-perm 0`（默认）与**没有这个功能时逐字相同**，
    所有已报数的臂不受影响。

    🔴 本函数只重排**记录**，绝不碰 `_parse_export_prompt`：置换后的记录仍走
    `parse_export_record` 那条冻结通路，所以里面每一道断言（`答案不在选项集`、
    `符号模式的 choice 类名必须是单个小写字母`、`score 档位编号不连续`）都自动成了
    本函数的安全网 —— 写错了会 FATAL，不会静默把 gold 配到别的选项上。
    """
    if not seed:
        return rec, None
    if rec.get('answer_format') != 'symbol':
        return rec, None
    q = rec['question']
    if 'Options:\n' not in q:
        return rec, None
    head, rest = q.split('Options:\n', 1)
    lines = rest.split('\n')
    ti = next((i for i, l in enumerate(lines) if l.startswith('Reply with exactly')), None)
    if ti is None:
        raise ValueError(f'{rec["id"]}: Options 块后面没有 "Reply with exactly" 尾行')
    opt_lines = lines[:ti]
    if any(not l.strip() for l in opt_lines):
        raise ValueError(f'{rec["id"]}: Options 块里有空行 —— 重排会吃掉它，题面就不逐字了')
    parsed = []
    for l in opt_lines:
        k, sep, d = l.partition(': ')
        if not sep or not re.fullmatch(r'[A-Za-z]', k):
            raise ValueError(f'{rec["id"]}: 选项行不认识（不是「单字母: 描述」）：{l!r}')
        parsed.append((k, d))
    k_n = len(parsed)
    if k_n < 2:
        return rec, None
    qid = rec['id'].split('::', 1)[1]
    perm = option_permutation(qid, seed, k_n)
    if perm == tuple(range(k_n)):
        return rec, None

    keys = [k for k, _ in parsed]
    descs = [d for _, d in parsed]
    gold = rec['answer'].strip()
    if gold not in keys:
        raise ValueError(f'{rec["id"]}: 答案 {gold!r} 不在选项集 {keys}')
    new_gold = keys[perm.index(keys.index(gold))]
    new_q = head + 'Options:\n' + '\n'.join(
        [f'{keys[j]}: {descs[perm[j]]}' for j in range(k_n)] + lines[ti:])

    out = dict(rec)
    out['question'] = new_q
    out['answer'] = new_gold
    sk = rec.get('symbol_keys')
    if sk is not None:
        if len(sk) != k_n:
            raise ValueError(f'{rec["id"]}: symbol_keys 有 {len(sk)} 项，选项有 {k_n} 个')
        out['symbol_keys'] = [sk[perm[j]] for j in range(k_n)]
    return out, perm


def parse_export_record(rec, slices=None, no_count=False):
    """`train_val.json`（messages）或 `test.json`（question/answer）一条 -> 统一行。

    `answer_format == 'symbol'` 时答案段是裸符号（choice 小写字母 / score 1–5 /
    noul a/b），题面是 `a: <text>` 的单 token 形式；老导出没有这个字段，仍按
    `Answer: …` 语义答案解析。符号模式在这里就断言 choice 的类名是单个小写字母、
    答案在类名里，装载时 `pack.encode_space` 再断言每个符号恰好 1 token。

    `slices` 是本臂的实际切片数（来自 sfrontend；None = 无切片流）；`no_count`
    见 `rewrite_prompt_slices` 的两条口径。训练与评估走同一处，不会一侧改一侧不改。
    """
    symbol_mode = rec.get('answer_format') == 'symbol'
    if 'messages' in rec:
        msgs = rec['messages']
        if [m.get('role') for m in msgs] != ['user', 'assistant']:
            raise ValueError(f'{rec.get("id")}: messages 必须是 user/assistant 两回合')
        user, answer = msgs[0]['content'], msgs[1]['content']
    else:
        user = '<image>\n' + rec['question']
        answer = rec['answer']
    prompt = rewrite_prompt_slices(_strip_image_placeholder(user), slices,
                                   no_count=no_count)
    kind, instr, extra = _parse_export_prompt(prompt)
    if symbol_mode:
        gold = answer.strip()
    else:
        m = _ANSWER_RE.match(answer.strip())
        if not m:
            raise ValueError(f'{rec.get("id")}: assistant/answer 不是 "Answer: …"：{answer!r}')
        gold = m.group(1).strip()
    if kind == 'noul':
        if symbol_mode:
            if gold not in ('a', 'b'):
                raise ValueError(f'{rec["id"]}: 符号 noul 答案 {gold!r} 只能是 a/b'
                                 '（a = No，b = Yes）')
            keys = ('a', 'b')
            target = {'noul': int(gold == 'b')}
            texts = keys
        else:
            if gold not in ('No', 'Yes'):
                raise ValueError(f'{rec["id"]}: noul 答案 {gold!r} 只能是 No/Yes')
            keys = ('No', 'Yes')
            target = {'noul': int(gold == 'Yes')}
            texts = tuple('Answer: ' + t for t in keys)
        ins = {'type': 'noul', 'instructions': instr}
        gold_text = gold
        source_texts = ('No', 'Yes') if symbol_mode else None
        single_token = symbol_mode
    elif kind == 'choice':
        opts = extra
        keys = tuple(k for k, _ in opts)
        if gold not in keys:
            raise ValueError(f'{rec["id"]}: 答案 {gold!r} 不在选项集 {keys}')
        target = {'choice': gold}
        ins = {'type': 'choice', 'instructions': instr, 'criteria': dict(opts)}
        gold_text = gold
        if symbol_mode:
            if not all(re.fullmatch(r'[a-z]', k) for k in keys):
                raise ValueError(f'{rec["id"]}: 符号模式的 choice 类名必须是单个小写字母：{keys}')
            texts = keys
            source_texts = tuple(desc for _, desc in opts)
            single_token = True
        else:
            texts = tuple('Answer: ' + k for k in keys)
            source_texts = None
            single_token = False
    else:
        levels = extra
        if symbol_mode:
            keys = tuple(str(i + 1) for i in range(len(levels)))
            if gold not in keys:
                raise ValueError(f'{rec["id"]}: 答案 {gold!r} 不在符号档位 {keys}')
            target = {'level': int(gold) - 1}
            texts = keys
            source_texts = tuple(levels)
            single_token = True
        else:
            keys = tuple(f'L{i}' for i in range(len(levels)))
            if gold not in levels:
                raise ValueError(f'{rec["id"]}: 答案 {gold!r} 不在档位 {levels}')
            target = {'level': levels.index(gold)}
            texts = tuple('Answer: ' + t for t in levels)
            source_texts = None
            single_token = False
        ins = {'type': 'score', 'instructions': instr, 'criteria': list(levels)}
        gold_text = gold
    space = pack.AnswerSpace(kind, tuple(keys), tuple(texts), kind == 'score',
                             source_texts=source_texts, single_token=single_token)
    sid, qid = rec['id'].split('::', 1)
    return {
        'id': rec['id'], 'study_id': sid, 'question': qid, 'image': rec['image'],
        'prompt': prompt, 'answer_text': answer.strip(), 'gold_text': gold_text,
        'kind': kind, 'space': space, 'target': target,
        'target_index': pack.target_index(space, target),
        'record': {'state': '', 'questions': {qid: ins}, 'targets': {qid: target}},
    }


def _export_prefix(qid):
    for p in ('finding_', 'score_', 'side_', 'organ_', 'acq_', 'prog_', 'triage_',
              'newq_'):
        if qid.startswith(p):
            return p
    return qid.split('_')[0]


def _load_export_annotations(keys):
    """只读名单的 `prefix/rule_class` 两列，供报告按 rule_class 切开；不参与选题。"""
    out = {}
    paths = (('question_keep.tsv', True),
             ('newq_keep.tsv', False))
    for rel, factory in paths:
        p = os.path.join(REPO, rel)
        if not os.path.isfile(p):
            continue
        with open(p, encoding='utf-8') as f:
            head = f.readline().rstrip('\n').split('\t')
            ix = {c: i for i, c in enumerate(head)}
            if not {'keep', 'study_id', 'question'} <= set(ix):
                continue
            for line in f:
                d = line.rstrip('\n').split('\t')
                if d[ix['keep']] != '1':
                    continue
                key = (d[ix['study_id']], d[ix['question']])
                if key not in keys or key in out:
                    continue
                sp = d[ix['split']] if 'split' in ix else ''
                if factory:
                    out[key] = (d[ix['prefix']], d[ix['rule_class']], sp)
                else:
                    fam = d[ix['family']] if 'family' in ix else _export_prefix(key[1])
                    out[key] = (fam, '', sp)
    return out

FINDING_RULES = ('asserted_positive', 'explicit_negative', 'hedged_positive',
                 'implied_negative', 'implied_negative_v2', 'negated_hedged',
                 'organ_backed_negative')


def _gate():
    """`analysis/gate.py` —— 「名单带着哪一档判据」的唯一实现（不在这里重写一份）。"""
    ana = os.path.join(REPO, 'analysis')
    if ana not in sys.path:
        sys.path.insert(0, ana)
    import gate as G
    return G


def _cache_meta(root, d):
    """<root>/<d>/meta.json 是否是一份编码器缓存（而不是清单/反向列表的元数据）。"""
    mp = os.path.join(root, d, 'meta.json')
    if not os.path.isfile(mp):
        return None
    try:
        with open(mp, encoding='utf-8') as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(meta.get('organs'), list) or not meta['organs']:
        return None
    if 'encoder_identity' not in meta:
        return None
    return meta


def resolve_cache(root=DEFAULT_CACHE, sha12=None):
    """缓存根 -> (编码器目录, meta)。认 meta.json 的内容，不认目录名的形状。"""
    if not os.path.isdir(root):
        raise SystemExit(
            f'RADAR 缓存根目录不存在：{root}\n'
            f'  stage 1（Part A 步骤 4）还没跑过 —— 先 `GO=1 bash run_radar_cache.sh`，'
            f'或换 `--cache-root` 指向已有缓存；`--feats zeros` 只能用来验流水线，不能报数。')
    if sha12 is None:
        cands = sorted(d for d in os.listdir(root)
                       if _cache_meta(root, d) is not None)
        if len(cands) != 1:
            raise SystemExit(
                f'{root} 下按 meta.json 认出的编码器目录有 {len(cands)} 个：{cands}\n'
                f'  必须恰好一个（两份不同权重混在同一根目录里，列语义就可能不是同一套）；'
                f'显式给 `--cache-sha12` 选一个。')
        sha12 = cands[0]
    d = os.path.join(root, sha12)
    mp = os.path.join(d, 'meta.json')
    if not os.path.isfile(mp):
        raise SystemExit(f'缺 {mp} —— 缓存目录不完整（a15 的身份文件），不能续跑也不能读')
    meta = _cache_meta(root, sha12)
    if meta is None:
        raise SystemExit(f'{mp} 不是编码器缓存 meta（缺 organs / encoder_identity）')
    organs = meta.get('organs')
    if not isinstance(organs, list) or not organs:
        raise SystemExit(f'{mp} 没有 organs 列 —— 特征行的定序语义不明，拒绝读')
    return d, meta


def load_keep_rows(keep_tsv=DEFAULT_KEEP_TSV, rules=None, newq_keep=DEFAULT_NEWQ_KEEP):
    """两份名单的 keep=1 行 -> 每行一个 `(split, sid, question, prefix, rule_class)`。

    `prefix`：出厂题 = 题名前缀（`finding_` / `score_` / `side_` / `organ_`）；
    新题 = 族名（`compartment` / …），记录在 `out_newq/questions/` 里。
    `rules`（必填，无默认）：只保留 `rule_class ∈ rules` 的 `finding_*` 行；
    其余前缀不受它管（它们各自有自己的判据，见 plan §8.1 与 C.26）。
    """
    if not rules:
        raise SystemExit('--rules 必须显式给（无隐式默认值，plan §8.1 是数据决策）：'
                         '逗号分隔，取值域 ' + '/'.join(FINDING_RULES))
    rules = tuple(rules)
    bad = [r for r in rules if r not in FINDING_RULES]
    if bad:
        raise SystemExit(f'--rules 里有未知规则 {bad}；取值域 ' + '/'.join(FINDING_RULES))

    G = _gate()
    rows, gate_seen, hist = [], set(), collections.Counter()
    with open(keep_tsv, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        if 'gate' not in head:
            raise SystemExit(f'{keep_tsv} 没有 `gate` 列 —— 这份名单不知道自己是用哪一档切的，'
                             f'不能吃（重跑 analysis/question_keep_list.py）')
        ix = {c: i for i, c in enumerate(head)}
        for line in f:
            d = line.rstrip('\n').split('\t')
            if d[ix['keep']] != '1':
                continue
            gate_seen.add(d[ix['gate']])
            pre, rc = d[ix['prefix']], d[ix['rule_class']]
            if pre == 'finding_':
                if rc not in rules:
                    continue
            elif rc:
                raise SystemExit(f'{keep_tsv} 里 {d[ix["question"]]} 的 prefix={pre} 却带着 '
                                 f'rule_class={rc} —— 前缀与规则列的对应坏了')
            rows.append((d[ix['split']], d[ix['study_id']], d[ix['question']], pre, rc))
            hist[pre + (rc or '')] += 1

    if gate_seen != {G.ADOPTED}:
        raise SystemExit(f'{keep_tsv} 是按 {sorted(gate_seen)} 切的，当前判据档位是 '
                         f'{G.ADOPTED} —— 换档要重跑 analysis/question_keep_list.py，'
                         f'不能让旧档的名单冒充新档')

    if not os.path.isfile(newq_keep):
        raise SystemExit(f'缺 {newq_keep} —— 五族新题名单（C.26），重跑 analysis/newq_allow_list.py')
    nq_gate = set()
    with open(newq_keep, encoding='utf-8') as f:
        head = f.readline().rstrip('\n').split('\t')
        if 'gate' not in head:
            raise SystemExit(f'{newq_keep} 没有 `gate` 列（同 keep TSV 的理由）')
        ix = {c: i for i, c in enumerate(head)}
        for line in f:
            d = line.rstrip('\n').split('\t')
            if d[ix['keep']] != '1':
                continue
            nq_gate.add(d[ix['gate']])
            fam = d[ix['family']]
            rows.append((d[ix['split']], d[ix['study_id']], d[ix['question']], fam, ''))
            hist[fam] += 1
    if nq_gate != {G.ADOPTED}:
        raise SystemExit(f'{newq_keep} 是按 {sorted(nq_gate)} 切的，当前判据档位是 {G.ADOPTED}'
                         f' —— 重跑 analysis/newq_allow_list.py')
    return rows, hist


VIS_TOKEN_MODES = ('all', 'organs', 'whole')


def select_vis_tokens(n_rows, n_organs, mode):
    """(缓存行数, 器官数, 模式) -> `(projector 行序的 npz 行号, pn, ph)`。

    三臂都挂在同一条基座（radar_ft 缓存、37 槽、oev、s32）上，只改「哪些行进序列」：
      `all`    = 主臂原样（36 器官 + 1 整图）；
      `organs` = A4「去掉整图 token」：只给 36 个器官槽，第 37 行不读也不进序列；
      `whole`  = A3「只用整图一个 token」：序列里 1 个视觉 token，projector 建在
                 `(n_organs=0, has_whole=True)` 上 ⇒ `n_tok=1`，第 37 行走到行号 0。
    `pn/ph` 是 projector 该用的器官数与整图标志（`n_vis = pn + ph`），model 与 dataset
    必须同一次推导，否则 `_check_rows` 那句「占位符数 == 槽数」在装配期就炸。

    🔴 **收窄必须真是收窄**：`organs`/`whole` 在没有整图行的缓存上（`n_rows == n_organs`，
    即 pretrain 那一支）与主臂逐位相同，却会挂上 `-vtorgans`/`-vtwhole` 的标签 ——
    那种假消融比没有消融更坏（读表的人以为它测过），所以当场拒。
    """
    mode = str(mode or 'all')
    if mode not in VIS_TOKEN_MODES:
        raise SystemExit(f'vis_tokens={mode!r} 只能是 {VIS_TOKEN_MODES}')
    n_rows, n_organs = int(n_rows), int(n_organs)
    if n_rows < n_organs or n_rows > n_organs + 1:
        raise SystemExit(f'行数 {n_rows} 与器官数 {n_organs} 对不上（只能是 N 或 N+1）')
    if mode == 'all':
        return tuple(range(n_rows)), n_organs, n_rows > n_organs
    if n_rows == n_organs:
        raise SystemExit(f'vis_tokens={mode} 在这份缓存上是空操作：它只有 {n_rows} 行、'
                         '没有整图行可去/可留 —— 用 all（A3/A4 都定义在 radar_ft 的 37 行上）')
    if mode == 'organs':
        return tuple(range(n_organs)), n_organs, False
    return (n_organs,), 0, True


def slot_permutation(study_id, seed, k):
    """`(study, seed) -> range(k) 的一个置换`；`seed == 0` ⇒ 恒等（跨进程稳定）。

    A1（去解剖索引）的第二半：36 个器官槽的**行序**按 study 打乱，配套的第一半是
    `OrganProjector(index=False)`（把身份嵌入换成一个共享向量）。缺任何一半都不是这条臂：
    只换共享向量而留着固定行序，第 20 行永远是肝，位置本身就是索引；只打乱行序而留着
    `nn.Embedding`，身份嵌入跟着位置走，模型学到的「位置 j 是谁」在每卷上都不一样 ——
    那是另一条臂（索引存在但错配），不在本次四臂里。

    键是 **study_id**：特征行、`organ_box`、`organ_zprof` 都是逐卷的，同一卷的所有题
    必须看到同一个槽序（否则 `has_feat`/region 与向量的对应会逐题漂）。
    🔴 用 sha256 不用内置 `hash()`（str 的 hash 带 PYTHONHASHSEED 随机盐）：分片 eval 是
    多进程，加了盐 ⇒ 同一卷在不同片上行序不同，那份数据合起来才错、每片自己自洽。
    置换只依赖 (study_id, seed) ⇒ 与批次、分片、epoch 顺序无关。
    """
    if int(seed) == 0:
        return tuple(range(k))
    h = hashlib.sha256(f'slot:{seed}:{study_id}'.encode('utf-8')).digest()
    rng = random.Random(int.from_bytes(h[:8], 'big'))
    p = list(range(k))
    rng.shuffle(p)
    return tuple(p)


class JevDataset(Dataset):
    """一个 split 的题集。构造时把记录整份读进内存（train 2.2 s / 0.58 GB，实测），
    每个 sample 现拼序列（`pack.build_example`，模板的唯一实现）。"""

    def __init__(self, tok, split, rules, *, keep_tsv=DEFAULT_KEEP_TSV,
                 newq_keep=DEFAULT_NEWQ_KEEP, newq_dir=DEFAULT_NEWQ_DIR,
                 cache_root=DEFAULT_CACHE, cache_sha12=None, feats='cache',
                 n_organs=None, only_prefixes=None, check_cache=True,
                 n_vis=None, prompt_only=False, frontend=None,
                 sfrontend=None, ph_id=None, organ_crop='off', crop_margin=0.1,
                 organ_map=None, need_feats=None, json_path=None, fusion_pool=0,
                 organ2d=False, organ_evidence=False, ev_ph_id=None,
                 organ_route=False, prompt_no_count=False, option_perm=0,
                 vis_tokens='all', slot_perm=0, organ_acquire=False):
        splits = tuple(split.split('+'))
        if not splits or any(s not in config.SPLITS for s in splits):
            raise SystemExit(f'split={split!r}：只能是 {config.SPLITS} 之一，或用 "+" 合并'
                             f'（例 train+val）')
        if len(set(splits)) != len(splits):
            raise SystemExit(f'split={split!r} 里 {splits} 有重复')
        if feats not in ('cache', 'live', 'zeros', 'none'):
            raise SystemExit(f'feats={feats!r} 只能是 cache（读 npz）/ live（冻结塔现算，'
                             '见 visual_frontend.py）/ zeros（冒烟）/ none（纯 2D：'
                             '不读任何 3D 特征，只喂宿主原生切片）')
        self.tok, self.split, self.splits = tok, split, splits
        self.json_path = os.path.abspath(json_path) if json_path else None
        self.rules = tuple(rules)
        self.feats_mode = feats
        self.frontend = frontend
        self.sfrontend = sfrontend
        self.organ_crop = str(organ_crop)
        if self.organ_crop not in ('off', 'z', 'bbox', 'mask'):
            raise SystemExit(f'organ_crop={organ_crop!r} 不在 off/z/bbox/mask')
        self.crop_margin = float(crop_margin)
        self.organ_map = organ_map
        self.organ_stats = collections.Counter()
        if self.organ_crop != 'off':
            if sfrontend is None:
                raise SystemExit('organ_crop != off 但没有切片前端 —— 裁剪臂要切片流'
                                 '（原生通道关掉时该旋钮无物可裁）')
            if not organ_map:
                raise SystemExit('organ_crop != off 但 organ_map 空 —— 先跑 '
                                 'analysis/organ_crop_map.py，把产出的表经 --organ-map 给进来')
        self.prompt_no_count = bool(prompt_no_count)
        self.option_perm = int(option_perm or 0)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if sfrontend is not None and ph_id is None:
            raise SystemExit('给了 sfrontend 却没给 ph_id：原生通道的图像插槽与器官插槽'
                             '必须用不同的 id，否则两次 masked_scatter 互相覆盖')
        self.ph_id = int(ph_id) if ph_id is not None else self.pad_id

        self.shard_dir, meta = (None, {})
        if feats == 'live':
            if frontend is None:
                if n_vis != 0:
                    raise SystemExit("feats=live 必须给 frontend=VisualFrontend(...)（loop 里按 "
                                     "--encoder 构造）—— 这里不自己建，是为了让 8 卡各 rank "
                                     '共用一份身份核对')
                self.n_organs = int(n_organs) if n_organs else RE.N_ORGANS
                self.n_rows = self.n_organs
                meta = {'visual_mode': 'live', 'encoder_identity': None,
                        'note': 'mode=state-only：未开塔，n_vis=0'}
            else:
                self.n_organs = int(n_organs) if n_organs else RE.N_ORGANS
                self.n_rows = int(frontend.n_rows)
                meta = frontend.describe()
                if self.n_rows != self.n_organs + (1 if frontend.want_whole else 0):
                    raise SystemExit(f'前端 n_rows={self.n_rows} 与身份 '
                                     f'{frontend.identity} 的 {self.n_organs}+whole 对不上')
        elif feats == 'cache':
            self.shard_dir, meta = resolve_cache(cache_root, cache_sha12)
            self.n_organs = len(meta['organs'])
            self.n_rows = self.n_organs + (1 if meta.get('has_whole') else 0)
            if n_organs and int(n_organs) != self.n_organs:
                raise SystemExit(f'n_organs={n_organs} != 缓存 meta 的 {self.n_organs}（a15）')
        elif feats == 'none':
            self.n_organs = int(n_organs) if n_organs else RE.N_ORGANS
            self.n_rows = self.n_organs
            meta = {'feats': 'none', 'note': '纯 2D：无 RADAR 特征、无器官槽、无融合'}
        else:
            if n_organs is None:
                raise SystemExit('feats=zeros 时必须显式给 n_organs（没有 meta 也没有前端，'
                                 '而这个数是特征行的真值来源）')
            self.n_organs = int(n_organs)
            self.n_rows = self.n_organs + (1 if getattr(frontend, 'want_whole', False) else 0)
        self.vis_tokens = str(vis_tokens or 'all')
        if self.feats_mode == 'none':
            if self.vis_tokens != 'all':
                raise SystemExit(f'feats=none（纯 2D）没有 3D 特征行，vis_tokens={self.vis_tokens} '
                                 '无物可选 —— 这条臂只有切片通道')
            self.vis_rows = tuple(range(self.n_rows))
            self.proj_n_organs, self.proj_has_whole = self.n_organs, self.n_rows > self.n_organs
        else:
            self.vis_rows, self.proj_n_organs, self.proj_has_whole = select_vis_tokens(
                self.n_rows, self.n_organs, self.vis_tokens)
        default_n_vis = 0 if self.feats_mode == 'none' else len(self.vis_rows)
        self.n_vis = default_n_vis if n_vis is None else int(n_vis)
        if self.feats_mode == 'none' and self.n_vis != 0:
            raise SystemExit(f'feats=none（纯 2D）只能配 n_vis=0，收到 {self.n_vis} —— '
                             '没有 3D 特征行可填器官槽')
        if self.n_vis == 0 and self.vis_tokens != 'all':
            raise SystemExit(f'n_vis=0（state-only / --organ-slots off / 纯 2D）配 '
                             f'vis_tokens={self.vis_tokens}：序列里一个 3D 槽都没有，'
                             '选哪几行没有意义')
        if self.n_vis not in (0, len(self.vis_rows)):
            raise SystemExit(f'n_vis={self.n_vis} 只能是 0（state-only / organ-slots off）'
                             f'或 {len(self.vis_rows)}（vis_tokens={self.vis_tokens} 的全部行）')
        self.fusion_pool = int(fusion_pool or 0)
        if self.fusion_pool:
            if sfrontend is None:
                raise SystemExit('fusion_pool>0 但没有切片前端 —— 池化的输入是切片 token')
            if self.fusion_pool != self.n_organs:
                raise SystemExit(f'fusion_pool={self.fusion_pool} 必须等于器官数 '
                                 f'{self.n_organs}（每个器官一个摘要 query）')
        self.organ_evidence = bool(organ_evidence)
        self.ev_ph_id = int(ev_ph_id) if ev_ph_id is not None else None
        if self.organ_evidence:
            if sfrontend is None:
                raise SystemExit('organ_evidence 需要切片前端 —— 证据 token 的 K/V '
                                 '来自宿主塔对切片的前向')
            if self.organ_crop == 'bbox':
                raise SystemExit('--organ-evidence 配 --organ-crop bbox 不合：平面裁剪改掉'
                                 '切片坐标系，organ_box 的区域归属就断了'
                                 '（off / z / mask 都保持全幅平面，可用）')
            if self.fusion_pool or organ2d:
                raise SystemExit('--organ-evidence 与 --fusion-pool/--organ2d 不合：'
                                 '三条臂都改器官段的序列/嵌入结构')
            if self.feats_mode == 'none':
                raise SystemExit('feats=none（纯 2D）不能配 organ_evidence：query 是 '
                                 'projector 器官嵌入，器官槽必须在序列里')
            if self.feats_mode == 'zeros':
                raise SystemExit('--organ-evidence 配 --feats zeros：没有 npz 就没有 '
                                 'organ_box，区域归属恒空 —— 只能验流水线，不许报数')
            if self.n_vis == 0:
                raise SystemExit('--organ-evidence 配 n_vis=0：器官槽不在序列里，'
                                 '证据 token 无槽可依')
            if not organ_map:
                raise SystemExit('organ_evidence 需要 organ_map（题名 -> 器官名，'
                                 'analysis/organ_crop_map.py）—— 先跑它再经 --organ-map 给进来')
            if self.ev_ph_id is None:
                raise SystemExit('organ_evidence 需要第二个空闲占位 id（ev_ph_id）—— '
                                 '由 loop 从 model.radar_slot_id 后取相邻空闲位传入')
            if self.ev_ph_id == self.ph_id:
                raise SystemExit(f'ev_ph_id {self.ev_ph_id} == 器官槽 ph_id —— '
                                 '两种插槽会互相覆盖')
        self.organ2d = bool(organ2d)
        if self.organ2d:
            if sfrontend is None:
                raise SystemExit('organ2d 需要切片前端 —— 2D 证据来自宿主塔对切片的前向')
            if self.organ_crop != 'off':
                raise SystemExit('--organ2d 与 --organ-crop 不合：裁剪改掉切片的平面坐标系，'
                                 'organ_box 的空间归属就断了（本臂的器官定位读 organ_box 原值）')
            if self.fusion_pool:
                raise SystemExit('--organ2d 与 --fusion-pool 不合：两者都改切片段的序列结构，'
                                 'organ2d 的口径是切片块整个不出现在序列里')
            if self.feats_mode == 'none':
                raise SystemExit('feats=none（纯 2D）不能配 organ2d：残差的基座是 projector '
                                 '器官槽，这条臂 3D 槽必须在序列里')
            if self.feats_mode == 'zeros':
                raise SystemExit('--organ2d 配 --feats zeros：zeros 没有 npz 也就没有 '
                                 'organ_box，聚合恒无效 —— 只能验流水线，不许报数')
            if self.n_vis == 0:
                raise SystemExit('--organ2d 配 n_vis=0：器官槽不在序列里，残差无处注入')
        self.organ_route = bool(organ_route)
        if self.organ_route:
            if self.n_vis == 0:
                raise SystemExit('--organ-route 配 n_vis=0：器官槽不在序列里，门控无槽可缩放')
            if not organ_map:
                raise SystemExit('organ_route 需要 organ_map（题名 -> 器官名，'
                                 'analysis/organ_crop_map.py）—— 先跑它再经 --organ-map 给进来')
        self.organ_acquire = bool(organ_acquire)
        if self.organ_acquire:
            if not self.organ_evidence:
                raise SystemExit('--organ-acquire 需要 --organ-evidence（获取动作掩的是证据槽）')
            if self.n_vis != self.n_organs + 1:
                raise SystemExit('--organ-acquire 要整图槽（n_vis==n_organs+1）：整图是可获取'
                                 '动作之一，没有第 37 槽这条臂的动作空间就缺了一格')
            if not organ_map:
                raise SystemExit('organ_acquire 需要 organ_map（题绑定 -> route_mask 的获取偏好）')
        if self.vis_tokens == 'whole':
            if sfrontend is not None:
                raise SystemExit('vis_tokens=whole 与切片通道不合：A3 的口径是序列里**只有**'
                                 '那一个整图 token —— 给它 --slices 0')
            if self.organ_evidence or self.organ2d:
                raise SystemExit('vis_tokens=whole 与 --organ-evidence/--organ2d 不合：两者的'
                                 ' K/V 都是切片 token，留着就不再是「一个 token」')
            if self.fusion_pool:
                raise SystemExit('vis_tokens=whole 与 --fusion-pool 不合：池化摘要同样是额外的'
                                 '视觉占位符')
            if self.organ_route:
                raise SystemExit('vis_tokens=whole 与 --organ-route 不合：路由缩放作用在器官槽上，'
                                 '这条臂没有器官槽')
        self.slot_perm = int(slot_perm or 0)
        self._perm_cache = collections.OrderedDict()
        if self.slot_perm and self.feats_mode == 'zeros':
            raise SystemExit('--slot-perm 配 --feats zeros：置换的是真特征行的顺序，'
                             'zeros 里没有顺序可言 —— 只能验流水线，不许报数')
        if self.slot_perm and len(self.vis_rows) < 2:
            raise SystemExit(f'--slot-perm 在 vis_tokens={self.vis_tokens} 上是空操作：'
                             f'序列里只有 {len(self.vis_rows)} 个 3D 槽，置换不了任何东西')
        self.need_feats = (self.n_vis > 0) if need_feats is None else bool(need_feats)
        if self.feats_mode == 'none' and self.need_feats:
            raise SystemExit('feats=none（纯 2D）不能需要特征行：融合/器官槽都必须关'
                             '（need_feats=True 是调用方接线错误）')
        self.prompt_only = bool(prompt_only)
        self.cache_root, self.cache_sha12 = cache_root, cache_sha12
        self.meta = meta
        self.infer_form = str(meta.get('infer_form') or RE.DEFAULT_INFER_FORM) \
            if meta else RE.DEFAULT_INFER_FORM
        self.geom_frame = str(meta.get('geom_frame') or
                              ('window' if self.infer_form == 'window' else 'nowin'))

        self.spaces, self.encs = {}, {}
        if self.json_path:
            self._init_json(only_prefixes)
        else:
            rows, hist = load_keep_rows(keep_tsv, rules, newq_keep)
            if only_prefixes:
                keep = set(only_prefixes)
                rows = [r for r in rows if r[3] in keep]
            rows = [r for r in rows if r[0] in self.splits]

            groups = collections.OrderedDict()
            seen = set()
            for sp, sid, q, pre, rc in rows:
                if (sid, q) in seen:
                    raise SystemExit(f'名单里 (study, question) 重复：{(sid, q)}')
                seen.add((sid, q))
                groups.setdefault(sid, []).append((sp, sid, q, pre, rc))
            self.study_order = list(groups)
            self.study_split = {}
            for sid, rs in groups.items():
                sps = {r[0] for r in rs}
                if len(sps) != 1:
                    raise SystemExit(f'study {sid} 同时出现在 {sorted(sps)} —— 出厂 split 不是逐卷唯一的，'
                                     f'合并集会让「谁见过这卷」说不清')
                self.study_split[sid] = rs[0][0]
            self.study_slices = []
            self.examples = []
            for sid, rs in groups.items():
                a = len(self.examples)
                self.examples.extend(rs)
                self.study_slices.append((a, len(self.examples)))
            if not self.examples:
                raise SystemExit(f'{split} 在 rules={self.rules} 下没有任何题 —— 旋钮给窄了？')

            self._records = self._load_records()
        self._rec_cache, self._feat_cache = collections.OrderedDict(), collections.OrderedDict()
        self._box_cache = collections.OrderedDict()
        self._zprof_cache = collections.OrderedDict()
        self.tag_hist = collections.Counter(
            (r[3] + (r[4] or '')) for r in self.examples)
        if check_cache and feats == 'cache':
            miss = self.missing_shards()
            if miss:
                raise SystemExit(
                    f'{split} 有 {len(miss)} 个 study 的嵌入不在缓存里（例：{miss[:3]}）\n'
                    f'  先跑 stage 1：`GO=1 bash run_radar_cache.sh`（Part A 步骤 4，'
                    f'当前 {self.shard_dir}）')

    def _load_records(self):
        """出厂记录（`out_v3/{split}.jsonl`）与新题记录（`out_newq/questions/{split}.jsonl`）
        各读一遍，按 study_id 归并。key 是 (source, study_id)，source ∈ {'factory','newq'}。
        合并集（`train+val`）= 每个成员 split 的两个文件都读。"""
        paths = {}
        for sp in self.splits:
            paths[('factory', sp)] = os.path.join(config.OUT, f'{sp}.jsonl')
            paths[('newq', sp)] = os.path.join(DEFAULT_NEWQ_DIR, f'{sp}.jsonl')
        recs = {}
        for (src, sp), path in paths.items():
            if not os.path.isfile(path):
                raise SystemExit(f'缺记录文件 {path}')
            with open(path, encoding='utf-8') as f:
                for ln, line in enumerate(f, 1):
                    r = json.loads(line)
                    if r.get('split') != sp:
                        raise SystemExit(f'{path}:{ln} split 字段是 {r.get("split")!r}，'
                                         f'文件却按 {sp} 载入')
                    sid = r['x_radev']['study_id']
                    if (src, sid) in recs:
                        raise SystemExit(f'{path}:{ln} study {sid} 重复')
                    recs[(src, sid)] = r
        return recs

    def _resolve_json_split(self, key, ann):
        """(study, question) -> 它自己的真实 split。

        `train+val.json` 不带 split 列，split 由 keep 名单的 `split` 列定（导出就是从
        那份名单切的）。名单只覆盖 keep=1 的行；导出集里出现名单外的 key 说明两份产物
        不同源。只有单 split 文件（test.json）允许按本次 `split` 兜底。
        """
        if ann and ann[2] in config.SPLITS:
            return ann[2]
        if len(self.splits) == 1:
            return self.splits[0]
        raise SystemExit(
            f'{self.json_path} 里的 {(key[0], key[1])} 不在一份 keep 名单里，而文件装载的是 '
            f'{self.splits} —— 没有 split 列就不知道该读 train/ 还是 val/ 的 npz。'
            '检查导出链与 out_inventory_prose_llm/question_keep.tsv 是否同源。')

    def _init_json(self, only_prefixes):
        """导出 `train_val.json` / `test.json` -> 与 test 逐字同源的训练序列。

        题面直接取导出文件的 user 内容（Vision-Jev 形式），不再用
        `state + instructions + answer_hint` 另拼一份 —— 训练序列与 `score_exact()`
        的候选前缀因此是同一串 token（导出端的 PROMPT_SHA 守着一份冻结模板）。
        `--rules` 照常生效：`finding_*` 行按名单的 `rule_class` 过滤，其余前缀不受管。
        """
        if not self.rules:
            raise SystemExit('导出 JSON 模式同样要 --rules（无隐式默认值，plan §8.1）')
        bad = [r for r in self.rules if r not in FINDING_RULES]
        if bad:
            raise SystemExit(f'--rules 里有未知规则 {bad}；取值域 ' + '/'.join(FINDING_RULES))
        if not os.path.isfile(self.json_path):
            raise SystemExit(f'缺导出数据 {self.json_path} —— 先跑 run_prose_finalize.sh')
        with open(self.json_path, encoding='utf-8') as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise SystemExit(f'{self.json_path} 不是 JSON 数组（导出链的产物是 pretty-printed 数组）')
        rows, seen = [], set()
        n_slices = getattr(self.sfrontend, 'n_slices', None)
        n_perm = 0
        for i, rec in enumerate(raw):
            try:
                rec, perm = apply_option_permutation(rec, self.option_perm)
                n_perm += 1 if perm is not None else 0
                row = parse_export_record(rec, slices=n_slices,
                                          no_count=self.prompt_no_count)
            except (KeyError, ValueError) as e:
                raise SystemExit(f'{self.json_path} 第 {i} 条解析失败：{e}')
            key = (row['study_id'], row['question'])
            if key in seen:
                raise SystemExit(f'{self.json_path} 第 {i} 条 (study, question) 重复：{key}')
            seen.add(key)
            rows.append((key, row))
        del raw
        if self.option_perm:
            print(f'[data] --option-perm={self.option_perm}：{n_perm} 道 choice 题的选项被轮转'
                  f'（共 {sum(1 for _, r in rows if r["kind"] == "choice")} 道 choice；'
                  'noul/score 一律不动，score 的档位序是语义）', flush=True)
        ann = _load_export_annotations(seen)

        groups = collections.OrderedDict()
        qspace = {}
        only = set(only_prefixes) if only_prefixes else None
        for key, row in rows:
            a = ann.get(key)
            split = self._resolve_json_split(key, a)
            if split not in self.splits:
                continue
            prefix = a[0] if a else _export_prefix(key[1])
            rc = a[1] if a else ''
            if prefix == 'finding_' and rc not in self.rules:
                continue
            if only is not None and prefix not in only:
                continue
            old = qspace.get(key[1])
            if old is None:
                qspace[key[1]] = (row['space'], (row['space'].kind, key[1]))
            elif (old[0].kind != row['space'].kind
                  or old[0].keys != row['space'].keys
                  or old[0].texts != row['space'].texts
                  or old[0].source_texts != row['space'].source_texts
                  or old[0].single_token != row['space'].single_token):
                raise SystemExit(
                    f'{key[1]} 在不同卷上的答案空间不一致（{old[0]} vs {row["space"]}）—— '
                    '同一题名的候选集必须稳定，否则受限 softmax 的类名与索引不可比')
            groups.setdefault(key[0], []).append((split, key[0], key[1], prefix, rc))
        if not groups:
            raise SystemExit(
                f'{self.json_path} 在 split={self.split} rules={self.rules} '
                f'only_prefixes={only_prefixes} 下没有任何题 —— 旋钮给窄了？')

        self._export_rows = dict(rows)
        self._json_qspace = qspace
        self.study_order = list(groups)
        self.study_split = {}
        self.study_slices = []
        self.examples = []
        for sid, rs in groups.items():
            sps = {r[0] for r in rs}
            if len(sps) != 1:
                raise SystemExit(f'study {sid} 同时出现在 {sorted(sps)} —— '
                                 'split 不是逐卷唯一的，合并集会让「谁见过这卷」说不清')
            self.study_split[sid] = rs[0][0]
            a0 = len(self.examples)
            self.examples.extend(rs)
            self.study_slices.append((a0, len(self.examples)))
        for qid, (space, key) in qspace.items():
            self.spaces[key] = space
            self.encs[key] = pack.encode_space(space, self.tok, lead='')

    def _source(self, pre):
        return 'factory' if pre.endswith('_') else 'newq'

    def _record(self, ex):
        if self.json_path:
            return self._export_rows[(ex[1], ex[2])]['record']
        key = (self._source(ex[3]), ex[1])
        r = self._rec_cache.get(key)
        if r is None:
            r = self._records.get(key)
            if r is None:
                raise SystemExit(f'{key} 在记录文件里不存在 —— 名单与出厂数据不同源')
            if not (ex[2] in r['questions'] and ex[2] in r['targets']):
                raise SystemExit(f'{ex[2]} 不在 {key} 的 questions/targets 里 —— 同上')
            self._rec_cache[key] = r
            if len(self._rec_cache) > 4:
                self._rec_cache.popitem(last=False)
        return r

    def _sp(self, sid):
        """这卷自己的真实 split —— 合并集下文件名/npz 目录都由它定，不由 `self.split` 定。"""
        sp = self.study_split.get(sid)
        if sp is None:
            raise SystemExit(f'study {sid} 不在本数据集的名单里，无从判断它属于哪个 split')
        return sp

    def _perm(self, sid):
        """该卷的 `(sel, inv)`：`sel[j]` = projector 第 j 行取自 npz 的哪一行，
        `inv[npz 行]` = 它在 projector 里的行号（按器官名查表要走逆映射）。

        整图行**永远钉在最后一位**：projector 的行序里它排在器官行之后，
        `model._evidence_embeds` 的整图分支按 `k == n_organs` 认它；把它一起洗进置换
        就同时改了两件事（内容顺序 + 那条分支的判据），不是这条臂要测的东西。
        """
        vr = self.vis_rows
        if not self.slot_perm or len(vr) < 2:
            return vr, {r: j for j, r in enumerate(vr)}
        e = self._perm_cache.get(sid)
        if e is None:
            k = self.n_organs
            perm = slot_permutation(sid, self.slot_perm, k)
            sel = tuple(vr[i] for i in perm) + tuple(vr[k:])
            e = (sel, {r: j for j, r in enumerate(sel)})
            self._perm_cache[sid] = e
            if len(self._perm_cache) > 8:
                self._perm_cache.popitem(last=False)
        return e

    def _rows_of(self, sid, names):
        """器官名列表 -> **本轮槽序下**的行号（`slot_perm=0` 时逐字等于 `RE.ORGANS.index`）。

        证据 token / 路由 / 裁剪三处都从器官名出发查行号，它们查的必须是「这个器官的
        向量现在坐在第几行」，而不是 npz 里的行号 —— 两者只在关掉置换时重合。
        名字不在 `RE.ORGANS` 里是映射表坏了，宁可炸也不静默丢一个器官。
        """
        ks = [RE.ORGANS.index(o) for o in names]
        if not self.slot_perm:
            return ks
        _, inv = self._perm(sid)
        try:
            return [inv[k] for k in ks]
        except KeyError as e:
            raise SystemExit(f'器官行 {e} 不在本轮选中的特征行里（vis_tokens='
                             f'{self.vis_tokens}）—— 按器官名查槽位的臂没被关掉？）')

    def _feats(self, sid):
        if self.feats_mode == 'zeros' or not self.need_feats:
            n = len(self.vis_rows) if self.n_vis else self.n_rows
            return (torch.zeros(n, 256), torch.zeros(n, dtype=torch.bool))
        if self.feats_mode == 'live':
            f = self._feat_cache.get(sid)
            if f is None:
                f = self.frontend.feats(sid, self._sp(sid))
                self._feat_cache[sid] = f
                if len(self._feat_cache) > 4:
                    self._feat_cache.popitem(last=False)
            return self._select(f, sid)
        f = self._feat_cache.get(sid)
        if f is None:
            path = os.path.join(self.shard_dir, self._sp(sid), f'{sid}.npz')
            if not os.path.isfile(path):
                raise SystemExit(f'缺缓存 {path}（本 sample 恰好用到它）')
            feats, has = VF.read_npz_rows(path, want_whole=self.n_rows > self.n_organs,
                                         who=str(self.meta.get('encoder_identity') or 'cache'))
            f = (torch.from_numpy(feats.astype(np.float32)), torch.from_numpy(has.copy()))
            self._feat_cache[sid] = f
            if len(self._feat_cache) > 4:
                self._feat_cache.popitem(last=False)
        return self._select(f, sid)

    def _select(self, f, sid):
        """npz 原始行 -> 「本轮进序列的行、按本轮槽序」—— `vis_tokens` 与 `slot_perm` 的落地处。

        LRU 里存的始终是 npz 原序的整份（`want_whole` 读来的 n_rows 行），选择发生在读缓存
        之后：置换以 study 为键，缓存按 study 复用，两边都不必知道对方。恒等（`all` +
        `slot_perm=0`）时**原对象直接返回**，不做一次 gather —— 主臂与今天逐位相同，
        连浮点张量的内存布局都不变。
        """
        sel, _ = self._perm(sid)
        if len(sel) == f[0].shape[0] and all(r == j for j, r in enumerate(sel)):
            return f
        ix = torch.as_tensor(list(sel), dtype=torch.long)
        return (f[0][ix], f[1][ix])

    def shard_path(self, sid):
        if self.feats_mode == 'live':
            return self.frontend.path(self._sp(sid), sid)
        return os.path.join(self.shard_dir, self._sp(sid), f'{sid}.npz')

    def missing_shards(self):
        if self.feats_mode != 'cache' or not self.need_feats:
            return []
        return [sid for sid in self.study_order if not os.path.isfile(self.shard_path(sid))]

    def npz_probs(self, sid):
        """② 级地板（floor.py）用的官方 20 征象逐例概率 + 键名。"""
        if self.feats_mode != 'cache':
            raise SystemExit(
                f'feats={self.feats_mode} 这条路上没有概率列：现算只出特征，不带文本塔。'
                'radar_ft 的文本塔是英文的（词表 30522），中文 .pt 与它不同源 —— '
                '② 级地板要先从缓存 feats + 英文 prompt 的文本嵌入离线重算 probs（任务 #60），'
                '拿中文嵌入去配英文塔得到的是一列语义错位的数')
        with np.load(self.shard_path(sid)) as z:
            return tuple(str(k) for k in z['prob_keys']), z['probs']

    def _box_sel(self, sid):
        """`organ_box` / `organ_zprof` 的行序（只含器官行）—— 与 `_feats` 同一个置换。

        这两张表与特征行同源（同一卷、同一个器官），A1 打乱槽序时它们必须跟着走：
        证据 token 读的是「第 j 行的向量 + 第 j 行的 box」，两边不同步就等于给每个器官
        配了别人的位置 —— 那测的不再是索引，是几何错配。
        返回 `None` = 恒等（调用方不索引 ⇒ 主臂逐位不变）。
        """
        if not self.slot_perm or len(self.vis_rows) < 2:
            return None
        sel, _ = self._perm(sid)
        rows = tuple(r for r in sel if r < self.n_organs)
        if all(r == j for j, r in enumerate(rows)):
            return None
        return list(rows)

    def _boxes(self, sid):
        """该卷的 `organ_box` int16[36,6]；缺列（旧缓存）/缺文件 → None —— 读侧回退。

        cache：直接读 shard npz。live：先 `_feats(sid)`（触发 write-through 落同一 npz），
        再读盘 —— write-through 关着时读到 None，逐题回退并计数。zeros：没有 npz。
        None 不写进 LRU（下次仍判一次文件，代价是一个 stat；写死会掩盖后续落盘）。
        """
        b = self._box_cache.get(sid)
        if b is not None:
            return b
        if self.feats_mode == 'cache':
            path = self.shard_path(sid)
        elif self.feats_mode == 'live':
            self._feats(sid)
            path = self.frontend.path(self._sp(sid), sid)
        else:
            return None
        if not os.path.isfile(path):
            return None
        b = VF.read_npz_box(path)
        if b is not None:
            sel = self._box_sel(sid)
            if sel is not None:
                b = np.asarray(b)[sel]
            self._box_cache[sid] = b
            if len(self._box_cache) > 8:
                self._box_cache.popitem(last=False)
        return b

    def _zprof(self, sid):
        """该卷的 `organ_zprof` int32[36,D]；缺列（旧缓存）/缺文件 → None。

        与 `_boxes` 同款读侧回退：None 不写 LRU（下次仍判一次 stat），命中上限 8。
        """
        a = self._zprof_cache.get(sid)
        if a is not None:
            return a
        if self.feats_mode == 'cache':
            path = self.shard_path(sid)
        elif self.feats_mode == 'live':
            self._feats(sid)
            path = self.frontend.path(self._sp(sid), sid)
        else:
            return None
        if not os.path.isfile(path):
            return None
        a = VF.read_npz_zprof(path)
        if a is not None:
            sel = self._box_sel(sid)
            if sel is not None:
                a = np.asarray(a)[sel]
            self._zprof_cache[sid] = a
            if len(self._zprof_cache) > 8:
                self._zprof_cache.popitem(last=False)
        return a

    def _crop_box(self, sid, qid):
        """题 -> crop 定位元组或 None（fallback 逐题计数）。

        顺序：题名查表（no_row / no_organs）→ 卷的 organ_box（no_box，含 mask 里全无该类）
        → 并集 bbox（union_crop 管 margin/最小边）。`mask` 臂再读 organ_zprof，把绑定
        器官的逐层体素数并集作为第 7 位一起返回；旧缓存缺列 → 第 7 位 None（逐题回退
        z 口径：bbox 的 z 范围、平面全幅，计 mask_noprof）。organ_crop == 'off' 时本函数不被调用。
        """
        organs = self.organ_map.get(qid) if self.organ_map is not None else None
        if organs is None:
            self.organ_stats['no_row'] += 1
            return None
        if not organs:
            self.organ_stats['no_organs'] += 1
            return None
        rows = self._boxes(sid)
        if rows is None:
            self.organ_stats['no_box'] += 1
            return None
        box = union_crop(rows[self._rows_of(sid, organs)],
                         margin=self.crop_margin)
        if box is None:
            self.organ_stats['no_box'] += 1
            return None
        self.organ_stats['ok'] += 1
        if getattr(self, 'organ_crop', 'z') != 'mask':
            return box
        prof = self._zprof(sid)
        if prof is None:
            self.organ_stats['mask_noprof'] += 1
            return box + (None,)
        occ = prof[self._rows_of(sid, organs)].sum(axis=0, dtype=np.int64)
        return box + (occ,)

    def _space(self, question, ins):
        """a14 的跨 study 一致性：同一题名在不同 study 上必须给出同一个答案空间
        （题面由 criteria 生成，理论上恒定；写在这里是因为 `space_for` 只看单条记录）。

        导出 JSON 分支不重推：候选文本（尤其 score 的完整档位短语）只存在于题面里，
        `_init_json` 已按题名登记并核过跨卷一致，这里直接取。
        """
        if self.json_path:
            try:
                return self._json_qspace[question]
            except KeyError:
                raise SystemExit(f'{question} 不在导出 JSON 的答案空间表里 —— '
                                 '数据集与记录不同源')
        space = pack.space_for(question, ins)
        key = (space.kind,) + tuple(space.keys)
        old = self.spaces.get(key)
        if old is None:
            self.spaces[key] = space
            self.encs[key] = pack.encode_space(space, self.tok)
        return space, key

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ex = self.examples[i]
        rec = self._record(ex)
        ins = rec['questions'][ex[2]]
        space, key = self._space(ex[2], ins)
        ai = pack.target_index(space, rec['targets'][ex[2]])
        pv = grid = None
        counts = None
        layers = None
        ev = None
        feats, has = self._feats(ex[1])
        if self.sfrontend is not None:
            vol = (self._export_rows[(ex[1], ex[2])].get('image')
                   if self.json_path else None)
            crop = (self._crop_box(ex[1], ex[2]) if self.organ_crop != 'off' else None)
            if self.organ2d or self.organ_evidence:
                pv, grid, layers = self.sfrontend.pixels(ex[1], self._sp(ex[1]),
                                                         with_layers=True, vol=vol)
            else:
                pv, grid = self.sfrontend.pixels(ex[1], self._sp(ex[1]),
                                                 crop=crop, vol=vol)
            counts = pack.slice_counts(grid, self.sfrontend.merge)
            if self.organ_evidence:
                if self.organ_acquire:
                    ev = [o for o in range(self.n_organs) if bool(has[o])]
                    ev.append(self.n_organs)
                    self.organ_stats['oev_acquire'] += 1
                else:
                    organs = self.organ_map.get(ex[2])
                    if organs is None:
                        self.organ_stats['oev_no_row'] += 1
                    elif not organs:
                        self.organ_stats['oev_no_organs'] += 1
                    if organs:
                        ev = sorted(self._rows_of(ex[1], organs))
                    elif self.n_vis == self.n_organs + 1:
                        ev = [self.n_organs]
                        self.organ_stats['oev_whole_fallback'] += 1
                    else:
                        ev = []
                        self.organ_stats['oev_no_whole_slot'] += 1
        if self.json_path:
            row = self._export_rows[(ex[1], ex[2])]
            seq = pack.build_export_example(
                self.tok, row['prompt'], self.encs[key].cand_ids,
                None if self.prompt_only else ai, self.n_vis, self.ph_id,
                img_counts=None if (self.fusion_pool or self.organ2d
                                    or self.organ_evidence) else counts,
                img_ids=(None if (self.organ2d or self.organ_evidence) else
                         (self.sfrontend.img_ids if counts is not None else None)),
                n_summary=self.fusion_pool or None,
                ev_organs=ev, ev_ph_id=self.ev_ph_id)
        else:
            seq = pack.build_example(self.tok, rec['state'], ins['instructions'], space,
                                     self.encs[key], None if self.prompt_only else ai,
                                     self.n_vis, self.ph_id,
                                     img_counts=None if (self.fusion_pool or self.organ2d
                                                         or self.organ_evidence) else counts,
                                     img_ids=(None if (self.organ2d or self.organ_evidence) else
                                              (self.sfrontend.img_ids if counts is not None else None)),
                                     n_summary=self.fusion_pool or None,
                                     ev_organs=ev, ev_ph_id=self.ev_ph_id)
        rm = None
        if self.organ_route or self.organ_acquire:
            organs = self.organ_map.get(ex[2]) if self.organ_map is not None else None
            rm = torch.zeros(self.n_vis, dtype=torch.bool)
            if organs:
                rm[sorted(self._rows_of(ex[1], organs))] = True
                self.organ_stats['ort_ok'] += 1
            else:
                self.organ_stats['ort_no_binding'] += 1
        out = {
            'input_ids': torch.tensor(seq['input_ids'], dtype=torch.long),
            'labels': torch.tensor(seq['labels'], dtype=torch.long),
            'vis_mask': torch.tensor(seq['vis_mask'], dtype=torch.bool),
            'answer_start': seq['answer_start'],
            'answer_len': seq['answer_len'],
            'feats': feats,
            'has_feat': has,
            'cand_ids': self.encs[key].cand_ids,
            'space_key': key,
            'target': ai,
            'question': ex[2],
            'prefix': ex[3],
            'rule_class': ex[4],
            'split': ex[0],
            'study_id': ex[1],
        }
        if rm is not None:
            out['route_mask'] = rm
        if 'mm_token_type_ids' in seq:
            out['mm_token_type_ids'] = torch.tensor(seq['mm_token_type_ids'], dtype=torch.long)
        if pv is not None:
            out['pixel_values'] = pv
            out['image_grid_thw'] = grid
            out['n_img'] = int(grid.shape[0])
            out['tower_rows'] = int(sum(counts))
        if self.organ2d or self.organ_evidence:
            box = self._boxes(ex[1])
            if box is None:
                self.organ_stats['o2d_no_box' if self.organ2d else 'oev_no_box'] += 1
                out['organ_box'] = torch.full((self.n_organs, 6), -1, dtype=torch.long)
            else:
                if box.shape[0] != self.n_organs:
                    raise SystemExit(f'{ex[1]} 的 organ_box 有 {box.shape[0]} 行 != 器官数 '
                                     f'{self.n_organs} —— npz 与缓存 meta 不同源')
                out['organ_box'] = torch.as_tensor(np.asarray(box), dtype=torch.long)
            out['slice_z'] = torch.as_tensor(np.asarray(layers), dtype=torch.long)
        if self.organ_evidence:
            out['ev_mask'] = torch.tensor(seq['ev_mask'], dtype=torch.bool)
            out['ev_organs'] = list(seq['ev_organs'])
        return out

    def space_of(self, key):
        return self.spaces[key]

    def enc_of(self, key):
        return self.encs[key]

    def collate(self, batch):
        """pad 到 batch 内最长；`cand_ids` 保持 ragged（模型按样本各自打分）。

        B 支的像素是**拼接**而不是堆叠：`pixel_values` 的行数由每片的 patch 数定，各样本
        可以不同，堆不成方阵。拼接序 = (样本序, 片序)，与序列里图像占位符的行序一致，
        `model._embed` 的 `masked_scatter` 就按这个序填；`pix_rows`/`n_img` 让模型能把
        属于某个样本的那段切回来（评估路径逐候选复制像素时要用）。
        """
        m = max(len(b['input_ids']) for b in batch)
        n = len(batch)
        ids = torch.full((n, m), self.pad_id, dtype=torch.long)
        lab = torch.full((n, m), -100, dtype=torch.long)
        vis = torch.zeros((n, m), dtype=torch.bool)
        att = torch.zeros((n, m), dtype=torch.long)
        has_mm = 'mm_token_type_ids' in batch[0]
        mm = torch.zeros((n, m), dtype=torch.long) if has_mm else None
        for k, b in enumerate(batch):
            L = len(b['input_ids'])
            ids[k, :L] = b['input_ids']
            lab[k, :L] = b['labels']
            vis[k, :L] = b['vis_mask']
            att[k, :L] = 1
            if has_mm:
                mm[k, :L] = b['mm_token_type_ids']
        if has_mm and int(mm.max()) not in (0, 1):
            raise SystemExit(f'mm_token_type_ids 出现 {int(mm.max())} —— 宿主的 M-RoPE 只认 '
                             '0=文本 / 1=图像 / 2=视频，RADAR 槽按 0 走')
        out = {
            'input_ids': ids, 'labels': lab, 'vis_mask': vis, 'attn_mask': att,
            'feats': torch.stack([b['feats'] for b in batch]),
            'has_feat': torch.stack([b['has_feat'] for b in batch]),
            'answer_start': torch.tensor([b['answer_start'] for b in batch], dtype=torch.long),
            'answer_len': torch.tensor([b['answer_len'] for b in batch], dtype=torch.long),
            'target': torch.tensor([b['target'] for b in batch], dtype=torch.long),
            'cand_ids': [b['cand_ids'] for b in batch],
            'space_key': [b['space_key'] for b in batch],
            'question': [b['question'] for b in batch],
            'prefix': [b['prefix'] for b in batch],
            'rule_class': [b['rule_class'] for b in batch],
            'study_id': [b['study_id'] for b in batch],
        }
        if 'route_mask' in batch[0]:
            if any('route_mask' not in b for b in batch):
                raise SystemExit('batch 里 route_mask 半开半关 —— organ_route 的接线断了')
            out['route_mask'] = torch.stack([b['route_mask'] for b in batch])
        if has_mm:
            out['mm_token_type_ids'] = mm
        if 'pixel_values' in batch[0]:
            if any('pixel_values' not in b for b in batch):
                raise SystemExit('batch 里有的样本有 pixel_values 有的没有 —— 切片通路'
                                 '半开半关，混批的语义没人定义')
            out['pixel_values'] = torch.cat([b['pixel_values'] for b in batch])
            out['image_grid_thw'] = torch.cat([b['image_grid_thw'] for b in batch])
            out['pix_rows'] = torch.tensor([b['pixel_values'].shape[0] for b in batch],
                                           dtype=torch.long)
            out['n_img'] = torch.tensor([b['n_img'] for b in batch], dtype=torch.long)
            if 'tower_rows' in batch[0]:
                out['tower_rows'] = torch.tensor([b['tower_rows'] for b in batch],
                                                 dtype=torch.long)
            if 'ev_mask' in batch[0]:
                if any('ev_mask' not in b for b in batch):
                    raise SystemExit('batch 里 ev_mask 半开半关 —— organ_evidence 的接线断了')
                ev = torch.zeros((n, m), dtype=torch.bool)
                for k, b in enumerate(batch):
                    ev[k, :len(b['ev_mask'])] = b['ev_mask']
                out['ev_mask'] = ev
                out['ev_organs'] = [list(b['ev_organs']) for b in batch]
            if 'organ_box' in batch[0]:
                if any('organ_box' not in b for b in batch):
                    raise SystemExit('batch 里 organ_box 半开半关 —— organ2d 的接线断了')
                out['organ_box'] = torch.stack([b['organ_box'] for b in batch])
                zs = [b['slice_z'] for b in batch]
                if len({int(z.numel()) for z in zs}) != 1:
                    raise SystemExit(f'batch 里 slice_z 片数不一：'
                                     f'{[int(z.numel()) for z in zs]} —— organ2d 假定各样本同片数')
                out['slice_z'] = torch.stack(zs)
        return out

    def describe(self):
        d = {
            'split': self.split, 'splits': list(self.splits), 'rules': list(self.rules),
            'data_source': 'export_json' if self.json_path else 'keep_tsv',
            'json_path': self.json_path,
            'n_studies': len(self.study_order), 'n_examples': len(self.examples),
            'n_split_members': {sp: sum(1 for v in self.study_split.values() if v == sp)
                                for sp in self.splits},
            'n_spaces_seen': len(self.spaces), 'n_organs': self.n_organs,
            'n_rows': self.n_rows,
            'n_vis': self.n_vis, 'prompt_only': self.prompt_only,
            'vis_tokens': self.vis_tokens, 'vis_rows': list(self.vis_rows),
            'proj_n_organs': self.proj_n_organs, 'proj_has_whole': self.proj_has_whole,
            'slot_perm': self.slot_perm,
            'need_feats': self.need_feats,
            'fusion_pool': self.fusion_pool,
            'organ2d': self.organ2d,
            'organ_evidence': self.organ_evidence,
            'organ_route': self.organ_route,
            'organ_acquire': self.organ_acquire,
            'prompt_no_count': self.prompt_no_count,
            'option_perm': self.option_perm,
            'ev_ph_id': self.ev_ph_id,
            'ph_id': self.ph_id,
            'native_slices': (self.sfrontend.n_slices if self.sfrontend else 0),
            'organ_crop': self.organ_crop,
            'organ_stats': dict(self.organ_stats),
            'feats': self.feats_mode, 'cache_dir': self.shard_dir,
            'by_tag': dict(sorted(self.tag_hist.items())),
        }
        if self.feats_mode == 'live':
            d['cache_dir'] = self.meta.get('root')
            keys = ('encoder_identity', 'ckpt_sha256', 'n_tensors_visual', 'has_whole',
                    'infer_form', 'geom_frame',
                    'saw_radar_train', 'published_csv_anchor', 'text_lang',
                    'write_through', 'lru', 'note')
            d['visual_frontend'] = {k: self.meta[k] for k in keys if k in self.meta}
        elif 'encoder_identity' in self.meta:
            d['encoder_identity'] = self.meta['encoder_identity']
            d['infer_form'] = self.infer_form
            d['geom_frame'] = self.geom_frame
            if self.meta.get('geom_src'):
                d['geom_src'] = self.meta['geom_src']
        return d


class StudyShuffledSampler(Sampler):
    """study 级 shuffle：study 之间洗牌，study 内部保持原序（同一 study 的题连着出，
    记录与嵌入只读一次）。`set_epoch` 换种子；`shuffle=False` 供 eval 用（顺序=名单顺序）。

    DDP：`rank`/`world_size` > 1 时，洗牌后按 `order[rank::world_size]` 把 study 分给
    各 rank（study 级分片，同一 study 的题不跨卡）。每 epoch 各 rank 看到的 study 子集
    不同（洗牌种子相同 ⇒ 分片确定性一致）。

    🔴 study 级分片的样本数不相等；DDP 每个 optimizer step 都要全员 allreduce，
    短的 rank 先耗尽数据去 barrier、长的还在 backward ⇒ 600s 后 NCCL watchdog
    打死整个 job（2026-10-25 s4-2B 在 step ~6900 实测）。因此 world_size>1 时
    所有 rank 的 `__len__` 统一补到最长分片，短的 rank 从自己的分片头部循环重复
    补尾 —— 重复占 <1%，换全员步数一致。

    `set_skip(n)`：续跑时快进 n 条样本（在采样器层跳过，不加载任何数据）。"""

    def __init__(self, slices, seed=0, shuffle=True, n_studies_limit=0,
                 rank=0, world_size=1):
        self.slices = list(slices)
        if n_studies_limit:
            self.slices = self.slices[:int(n_studies_limit)]
        self.seed, self.shuffle, self.epoch = seed, shuffle, 0
        self.rank, self.world_size = int(rank), int(world_size)
        self._skip = 0

    def _order(self):
        order = list(range(len(self.slices)))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(order), generator=g).tolist()
        return order

    def _my_studies(self):
        order = self._order()
        if self.world_size > 1:
            order = order[self.rank::self.world_size]
        return order

    def _count(self, studies):
        return sum(self.slices[s][1] - self.slices[s][0] for s in studies)

    def _target_len(self):
        if self.world_size <= 1:
            return self._count(self._my_studies())
        return max(self._count(self._order()[r::self.world_size])
                   for r in range(self.world_size))

    def __len__(self):
        return self._target_len()

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def set_skip(self, n):
        """下一次迭代开始前跳过 n 条样本；只生效一次。"""
        self._skip = max(0, int(n))

    def _indices(self, target):
        studies = self._my_studies()
        n = 0
        while studies and n < target:
            for s in studies:
                a, b = self.slices[s]
                for i in range(a, b):
                    if n >= target:
                        return
                    n += 1
                    yield i

    def __iter__(self):
        it = self._indices(self._target_len())
        skip, self._skip = self._skip, 0
        if skip:
            it = itertools.islice(it, skip, None)
        yield from it
