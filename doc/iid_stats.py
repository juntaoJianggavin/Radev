import os
"""Compute IID dataset documentation stats from primary sources (read-only).

Sources: out_inventory/question_keep.tsv (keep gate), out_dataset_final_sym/*.json
(final training/eval set), out_v3/*.jsonl (study->split + patient_leakage flag).
Writes out_dataset_doc/iid_stats.json. No training/reported dirs touched.
"""
import csv, json, os, collections

ROOT = os.environ.get('RADEV_SOURCE_ROOT', '.')
OUT = os.path.join(ROOT, 'out_dataset_doc'); os.makedirs(OUT, exist_ok=True)
PY = None

def fam_of(qid):
    for p, f in (('finding_', 'finding'), ('organ_', 'organ'), ('score_', 'score'),
                 ('side_', 'side'), ('triage_', 'triage'), ('prog_', 'prog'), ('acq_', 'acq')):
        if qid.startswith(p):
            return f, qid[len(p):]
    return 'other', qid

def type_of(qid):
    f, _ = fam_of(qid)
    if f == 'finding': return 'noul'
    if f == 'score': return 'score'
    if f in ('organ', 'side', 'triage'): return 'choice'
    return 'other'

# ---------- A) keep gate ----------
rows = list(csv.DictReader(open(os.path.join(ROOT, 'out_inventory/question_keep.tsv'),
                                encoding='utf-8'), delimiter='\t'))
tot = len(rows)
keep1 = [r for r in rows if r['keep'] == '1']
keep0 = [r for r in rows if r['keep'] != '1']
A = {
    'factory_total': tot,
    'keep1': len(keep1), 'keep0': len(keep0),
    'drop_ratio': len(keep0) / tot,
    'keep1_by_split': dict(collections.Counter(r['split'] for r in keep1)),
    'keep0_by_split': dict(collections.Counter(r['split'] for r in keep0)),
    'factory_by_split': dict(collections.Counter(r['split'] for r in rows)),
    'keep1_by_prefix': dict(collections.Counter(r['prefix'] for r in keep1)),
    'keep1_by_rule_class': dict(collections.Counter(r['rule_class'] for r in keep1)),
    'keep0_by_why': dict(collections.Counter(r['why'] for r in keep0)),
    'gate_values': dict(collections.Counter(r['gate'] for r in rows)),
}
# study -> split uniqueness (study-level split check)
st_split = collections.defaultdict(set)
for r in rows:
    st_split[r['study_id']].add(r['split'])
multi = {s: sorted(v) for s, v in st_split.items() if len(v) > 1}
A['n_studies'] = len(st_split)
A['studies_in_multiple_splits'] = len(multi)
A['multi_split_examples'] = dict(list(multi.items())[:10])
A['studies_by_split'] = dict(collections.Counter(next(iter(v)) for v in st_split.values()))

# keep1 organ / disease distribution
org_keep = collections.Counter(); dis_keep = collections.Counter()
dis_pos = collections.Counter(); dis_neg = collections.Counter()
for r in keep1:
    f, key = fam_of(r['question'])
    if f == 'organ': org_keep[key] += 1
    elif f in ('finding', 'score', 'side'):
        dis_keep[key] += 1
# finding polarity pos/neg from rule_class
for r in keep1:
    if r['prefix'] == 'finding_':
        d = r['question'][len('finding_'):]
        rc = r['rule_class']
        if rc in ('asserted_positive', 'hedged_positive'): dis_pos[d] += 1
        else: dis_neg[d] += 1
A['keep1_organ_dist'] = dict(org_keep.most_common())
A['keep1_disease_dist'] = dict(dis_keep.most_common())
A['keep1_finding_pos'] = dict(dis_pos.most_common())
A['keep1_finding_neg'] = dict(dis_neg.most_common())

# ---------- B) final dataset ----------
study2split = {s: next(iter(v)) for s, v in st_split.items()}
def load_final(fn):
    return json.load(open(os.path.join(ROOT, 'out_dataset_final_sym', fn), encoding='utf-8'))
final = {'test': load_final('test.json'), 'train_val': load_final('train_val.json')}
B = {}
final_ids = set()
split_type = collections.defaultdict(collections.Counter)
split_fam = collections.defaultdict(collections.Counter)
split_org = collections.defaultdict(collections.Counter)
split_dis = collections.defaultdict(collections.Counter)
for fname, items in final.items():
    for it in items:
        sid, qid = it['id'].split('::', 1)
        sp = study2split.get(sid, 'UNKNOWN')
        if fname == 'test': sp = 'test'
        final_ids.add(it['id'])
        ty = it.get('type') or type_of(qid)
        if fname == 'test' and it.get('type') and it['type'] != type_of(qid):
            B.setdefault('type_mismatch', []).append(it['id'])
        split_type[sp][ty] += 1
        f, key = fam_of(qid)
        split_fam[sp][f] += 1
        if f == 'organ': split_org[sp][key] += 1
        elif f in ('finding', 'score', 'side'): split_dis[sp][key] += 1
B['final_total'] = sum(len(v) for v in final.values())
B['final_by_file'] = {k: len(v) for k, v in final.items()}
B['final_type_by_split'] = {k: dict(v) for k, v in split_type.items()}
B['final_fam_by_split'] = {k: dict(v) for k, v in split_fam.items()}
B['final_organ_by_split'] = {k: dict(v) for k, v in split_org.items()}
B['final_disease_by_split'] = {k: dict(v) for k, v in split_dis.items()}
# proper unique-study-per-split count
_ss = collections.defaultdict(set)
for fname, items in final.items():
    for it in items:
        sid = it['id'].split('::')[0]
        sp = 'test' if fname == 'test' else study2split.get(sid, 'UNKNOWN')
        _ss[sp].add(sid)
B['final_studies_by_split'] = {k: len(v) for k, v in _ss.items()}
# unique studies in final
B['final_unique_studies'] = len({it['id'].split('::')[0] for items in final.values() for it in items})

# ---------- C) keep1 vs final reconciliation ----------
keep1_ids = {r['study_id'] + '::' + r['question'] for r in keep1}
C = {
    'keep1_not_in_final': len(keep1_ids - final_ids),
    'final_not_in_keep1': len(final_ids - keep1_ids),
    'final_and_keep1': len(final_ids & keep1_ids),
}
drop_fam = collections.Counter(); drop_rule = collections.Counter()
for r in keep1:
    if r['study_id'] + '::' + r['question'] not in final_ids:
        drop_fam[r['prefix']] += 1; drop_rule[r['rule_class']] += 1
C['keep1_dropped_to_final_by_prefix'] = dict(drop_fam)
C['keep1_dropped_to_final_by_rule'] = dict(drop_rule)

# ---------- D) patient_leakage flag from out_v3 ----------
flag = collections.Counter(); flag_by_split = collections.defaultdict(collections.Counter)
for sp in ('train', 'val', 'test'):
    p = os.path.join(ROOT, 'out_v3', sp + '.jsonl')
    with open(p, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            x = r.get('x_radev', {})
            v = x.get('patient_leakage_possible')
            flag[str(v)] += 1; flag_by_split[sp][str(v)] += 1
D = {'patient_leakage_possible_counts': dict(flag),
     'by_split': {k: dict(v) for k, v in flag_by_split.items()}}

res = {'A_keep_gate': A, 'B_final': B, 'C_reconcile': C, 'D_leakage_flag': D}
json.dump(res, open(os.path.join(OUT, 'iid_stats.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
# compact console summary
print('FACTORY', tot, 'KEEP1', len(keep1), 'KEEP0', len(keep0), 'drop %.3f' % A['drop_ratio'])
print('keep1_by_split', A['keep1_by_split'])
print('studies', A['n_studies'], 'multi_split', A['studies_in_multiple_splits'], 'by_split', A['studies_by_split'])
print('FINAL total', B['final_total'], 'by_file', B['final_by_file'])
print('final_type_by_split', json.dumps(B['final_type_by_split']))
print('final_fam_by_split', json.dumps(B['final_fam_by_split']))
print('final_unique_studies', B['final_unique_studies'])
print('RECONCILE', json.dumps(C))
print('LEAK_FLAG', json.dumps(D))