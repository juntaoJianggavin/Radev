# organ_crop_map 覆盖报告

生成：`analysis/organ_crop_map.py`（幂等，可重跑）。
map.tsv：241 个题名（四族并集；acq_/prog_/triage_ 无绑定不收录 → 读侧 no_row）。
分母：`question_keep.tsv` 全行 733,452；其中 keep=1 390,974。

口径：命中 = 题名有 map 行且组有器官（crop 可用）；无类目 = 组在 RADAR 无类目
（5 空组）→ crop=None 回退 off 臂；无行 = 题名不在 map（acq_/prog_/triage_/未登记）。

## 逐组覆盖（组 = 16 组之一；行数以 keep=1 为主）

| 组 | 器官数 | 全行 | keep=1 行 | 命中(keep) | 无类目(keep) |
|---|---|---|---|---|---|
| Lower thorax | 7 | 94,060 | 53,001 | 53,001 | 0 |
| Liver and biliary tree | 2 | 84,683 | 39,453 | 39,453 | 0 |
| Gallbladder | 1 | 48,334 | 40,769 | 40,769 | 0 |
| Spleen | 1 | 40,946 | 21,708 | 21,708 | 0 |
| Pancreas | 1 | 41,192 | 38,689 | 38,689 | 0 |
| Adrenal glands | 1 | 0 | 0 | 0 | 0 |
| Kidneys and ureters | 1 | 63,815 | 48,028 | 48,028 | 0 |
| Gastrointestinal tract | 5 | 67,911 | 28,628 | 28,628 | 0 |
| Lymph nodes | 0 | 37,699 | 21,123 | 0 | 21,123 |
| Vasculature | 6 | 78,730 | 34,205 | 34,205 | 0 |
| Peritoneal cavity | 0 | 43,550 | 30,540 | 0 | 30,540 |
| Abdominal wall | 0 | 0 | 0 | 0 | 0 |
| Musculoskeletal | 12 | 37,796 | 27,035 | 27,035 | 0 |
| Bladder | 1 | 0 | 0 | 0 | 0 |
| Uterus and ovaries | 0 | 0 | 0 | 0 | 0 |
| Prostate and seminal vesicles | 0 | 15,544 | 7,795 | 0 | 7,795 |
| （无绑定：acq_/prog_/triage_/未登记 → no_row） | - | 79,192 | 0 | 0 | 0 |

**合计（keep=1）**：命中 331,516 / 无类目 59,458 / 无行 0；fallback 15.2%（对 keep=1）。
注：keep=1 名单当前不含 acq_/prog_/triage_（其 keep 恒 0），故 keep 口径无行 = 0；全行口径下有 79,192 行无绑定（读侧记 no_row）。

## 16 组 → RADAR 器官

| 组 | 器官数 | organs |
|---|---|---|
| Lower thorax | 7 | 心脏 肺 食管 肋骨 锁骨 胸椎 肺动脉 |
| Liver and biliary tree | 2 | 肝 门静脉 |
| Gallbladder | 1 | 胆囊 |
| Spleen | 1 | 脾 |
| Pancreas | 1 | 胰腺 |
| Adrenal glands | 1 | 肾上腺 |
| Kidneys and ureters | 1 | 肾 |
| Gastrointestinal tract | 5 | 食管 胃 小肠 大肠 十二指肠 |
| Lymph nodes | 0 | **（无类目→fallback）** |
| Vasculature | 6 | 主动脉 下腔静脉 髂动脉 髂静脉 门静脉 肺动脉 |
| Peritoneal cavity | 0 | **（无类目→fallback）** |
| Abdominal wall | 0 | **（无类目→fallback）** |
| Musculoskeletal | 12 | 竖脊肌 臀肌 髂腰肌 股骨 肱骨 肩胛骨 骶骨 颈椎 腰椎 胸椎 髋关节 锁骨 |
| Bladder | 1 | 膀胱 |
| Uterus and ovaries | 0 | **（无类目→fallback）** |
| Prostate and seminal vesicles | 0 | **（无类目→fallback）** |

## join 来源计数（audit 条目数）

- out_v3 三卷：finding_/score_ 424,521、organ_ 228,537、side_ 1,202（题名去重后 241）
- out_newq 两文件：newq_ 34,877
- 重跑：`<jjt_mllm>/bin/python analysis/organ_crop_map.py`

