import os
"""Final documentation stats for the ACTUAL train/test files:
   train+val = out_dataset_final_sym/train_val.json ; test = out_dataset_final_sym/test.json.
study->split from out_v3 (authoritative). Writes out_dataset_doc/final_stats.json.
"""
import json, os, collections

ROOT = os.environ.get('RADEV_SOURCE_ROOT', '.')
OUT = os.path.join(ROOT, 'out_dataset_doc')

def fam_of(qid):
    for p, f in (('finding_', 'finding'), ('organ_', 'organ'), ('score_', 'score'),
                 ('side_', 'side'), ('triage_', 'triage'), ('newq_', 'newq')):
        if qid.startswith(p):
            return f, qid[len(p):]
    return 'other', qid

def infer_type(user_text, ans):
    if ans in ('1', '2', '3', '4', '5'): return 'score'
    if 'a = No, b = Yes' in user_text: return 'noul'
    return 'choice'

# study -> split from out_v3
study2split = {}
for sp in ('train', 'val', 'test'):
    with open(os.path.join(ROOT, 'out_v3', sp + '.jsonl'), encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                study2split[r['x_radev']['study_id']] = sp

te = json.load(open(os.path.join(ROOT, 'out_dataset_final_sym/test.json'), encoding='utf-8'))
tv = json.load(open(os.path.join(ROOT, 'out_dataset_final_sym/train_val.json'), encoding='utf-8'))

# validate inference on test (has ground-truth 'type')
mism = 0
for it in te:
    if infer_type(it['question'], it['answer']) != it['type']:
        mism += 1
print('type-inference mismatches on test:', mism, '/', len(te))

agg = {sp: {'total': 0, 'type': collections.Counter(), 'fam': collections.Counter(),
            'organ': collections.Counter(), 'disease': collections.Counter(),
            'fd': collections.Counter(), 'fd_pos': collections.Counter(),
            'studies': set()} for sp in ('train', 'val', 'test', 'UNKNOWN')}

def add(sid, qid, ty, sp, ans):
    a = agg[sp]; a['total'] += 1; a['type'][ty] += 1
    f, key = fam_of(qid); a['fam'][f] += 1; a['studies'].add(sid)
    if f == 'organ': a['organ'][key] += 1
    elif f in ('score', 'side', 'newq'): a['disease'][key] += 1
    elif f == 'finding':
        a['fd'][key] += 1
        if ans == 'b': a['fd_pos'][key] += 1

for it in te:
    sid, qid = it['id'].split('::', 1)
    add(sid, qid, it['type'], 'test', it['answer'])
for it in tv:
    sid, qid = it['id'].split('::', 1)
    sp = study2split.get(sid, 'UNKNOWN')
    u = [m['content'] for m in it['messages'] if m['role'] == 'user'][0]
    a = [m['content'] for m in it['messages'] if m['role'] == 'assistant'][0]
    add(sid, qid, infer_type(u, a), sp, a)

res = {}
for sp, a in agg.items():
    res[sp] = {'total': a['total'], 'n_studies': len(a['studies']),
               'type': dict(a['type']), 'fam': dict(a['fam']),
               'organ': dict(a['organ'].most_common()),
               'finding_disease': dict(a['fd'].most_common()),
               'finding_pos': dict(a['fd_pos'].most_common()),
               'other_topic': dict(a['disease'].most_common())}
json.dump(res, open(os.path.join(OUT, 'final_stats.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

for sp in ('train', 'val', 'test', 'UNKNOWN'):
    a = res[sp]
    print('\n==', sp, 'total', a['total'], 'studies', a['n_studies'])
    print('  type', a['type'])
    print('  fam', a['fam'])
    print('  organ', a['organ'])
    print('  finding_pos_sum', sum(a['finding_pos'].values()),
          'finding_total', sum(a['finding_disease'].values()))
print('\nGRAND total', sum(res[s]['total'] for s in res))