# Radev 数据集文档（以实际 train / test 文件为准）

生成：2026-11-15。数字均由脚本从原始产物直接统计：
`out_dataset_doc/final_stats.py` → `final_stats.json`（实际 train/test），
`iid_stats.py` → `iid_stats.json`（清洗闸），审计见 `out_iid_audit400/`。

**训练集 = `out_dataset_final_sym/train_val.json`（train+val，223,286）；
测试集 = `out_dataset_final_sym/test.json`（56,825）。合计 280,111。**
（`out_dataset_final/` 为同内容文本答案版，`_sym` 为符号答案 a/b/1–5 版。）

---

## 1. 数据划分细节 + 每 split 样本量 + 泄漏（需求 1 / 8 / 9）

- **来源**：IID = the source cohort 腹部 CT（`reports_final.xlsx`+`metadata.csv`+`five_years_disease_task.csv`），
  图像 `iid/<study_id>.nii.gz`；labeler=rule_v4，schema 2.0，route t2。
- **划分单位 = study（study_id）**，split 继承源数据 `Split` 列（非重新随机）。
- **已在实际文件上验证**：train_val 与 test **study 重叠=0、image 重叠=0** ⇒ **无同 study/同报告/同图像泄漏**。
- **⚠️ 不是 patient 级**：无 patient_id，同一病人不同 study 可能跨 split；源数据对全部 study 打
  `patient_leakage_possible=True`。**只能声称防同 study/同报告，不能声称防同病人。**
- `02_leakage` 断言（state_leak=0 / state_acq_leak=0）查的是**题面是否泄答案**，另一回事，已通过。

**每 split（实际文件）：**

| split | 问题数 | study 数 |
|---|---|---|
| train | 168,093 | 15,178 |
| val | 55,175 | 5,009 |
| test | 56,825 | 5,080 |
| 合计 | **280,111** | **25,270** |

（train_val 里另有 18 条 / 3 study 不在源 split 映射中，属 few-shot/边缘条目，计入 train_val 223,286。）

---

## 2. Question type 分布（需求 2）

**按答案类型（实际文件，test 自带 type，train_val 按题面推断并在 test 上 0 误差验证）：**

| split | noul(Yes/No) | choice(MC) | score(L0–L4) | 合计 |
|---|---|---|---|---|
| train | 112,322 | 47,798 | 7,973 | 168,093 |
| val | 36,926 | 15,702 | 2,547 | 55,175 |
| test | 37,925 | 16,210 | 2,690 | 56,825 |
| 合计 | 187,189 | 79,711 | 13,211 | 280,111 |

**按题族：** finding 174,093 / organ 60,014 / **newq 26,553** / score 13,211 / triage 4,997 / side 1,243。
- finding=极性 noul；organ=器官最显著异常 MC；score=严重度；side=左右；triage=急症；
  newq=后加新题族（141 唯一题名，如 diverticulum/wall_change/extent/hernia_site/acute_chronic，含 noul+choice）。
- prog_ / acq_ 整族不保留。

---

## 3. Organ / Disease 分布（需求 3）

**Organ（实际文件只剩 8 个；spleen/pancreas/lymph/prostate 四个「退化 2 选项」器官整族移除）：**
liver 12,683 / lower(胸腔下部) 9,543 / gastrointestinal 8,140 / kidneys 7,842 /
musculoskeletal 6,981 / peritoneal 6,398 / vasculature 5,486 / gallbladder 2,940。

**Finding（30 病种，合计 174,077；pos 65,184 = 37.4% / neg 108,893）：**

| disease | tot | pos | neg | | disease | tot | pos | neg |
|---|---|---|---|---|---|---|---|---|
| gallstones | 15,085 | 1,985 | 13,100 | | free_air | 5,997 | 579 | 5,418 |
| atelectasis | 12,271 | 11,720 | 551 | | abdominal_aortic_aneurysm | 5,784 | 219 | 5,565 |
| renal_hypodensities | 11,929 | 5,630 | 6,299 | | hepatic_steatosis | 4,898 | 2,646 | 2,252 |
| surgically_absent_gallbladder | 11,689 | 577 | 11,112 | | lymphadenopathy | 4,856 | 979 | 3,877 |
| bowel_obstruction | 11,297 | 564 | 10,733 | | ascites | 3,476 | 2,577 | 899 |
| hydronephrosis | 10,615 | 935 | 9,680 | | fracture | 3,194 | 1,345 | 1,849 |
| biliary_ductal_dilation | 10,040 | 2,625 | 7,415 | | submucosal_edema | 3,126 | 1,667 | 1,459 |
| renal_cyst | 9,247 | 8,086 | 1,161 | | appendicitis | 2,956 | 133 | 2,823 |
| metastatic_disease | 8,659 | 792 | 7,867 | | pancreatic_atrophy | 2,933 | 1,695 | 1,238 |
| anasarca | 8,409 | 1,465 | 6,944 | | cardiomegaly | 1,868 | 792 | 1,076 |
| pleural_effusion | 8,160 | 3,809 | 4,351 | | coronary_calcification | 1,776 | 1,644 | 132 |
| atherosclerosis | 7,656 | 7,264 | 392 | | hiatal_hernia | 1,690 | 1,650 | 40 |
| | | | | | osteopenia | 1,627 | 1,023 | 604 |
| | | | | | splenomegaly | 1,601 | 876 | 725 |
| | | | | | hepatomegaly | 1,467 | 425 | 1,042 |
| | | | | | thrombosis | 869 | 707 | 162 |
| | | | | | prostatomegaly | 659 | 650 | 9 |
| | | | | | aortic_valve_calcification | 243 | 125 | 118 |

（逐 split 的 organ/disease/pos 明细见 `final_stats.json`。）

---

## 4. Label verification 流程（需求 4）

1. **规则打标**（rule_v4）：从报告器官段抽 finding 极性 / organ 最显著异常 / score / side / triage。出厂 733,452 题。
2. **双基座 LLM 盲审**：qwen32b=Qwen3-32B、phi4=Phi-4，对去重后的器官段文本单元（tid）独立投票。
3. **第三读者仲裁**：qwen38=**Qwen3.8-27B**（不是 8B），仅对「一票 S」平票集补判。
4. **Gate V2 = 三读者 ≥2 票 Support** 才保留。organ 自 2026-10-12 吃自己的三票；triage 2026-10-15 并入。
5. **implied negative 也过三读者**：「没提⇒没有」的隐含阴性用**中性模板**重问（避免诱导性 yes/no），
   同样由 qwen32b/phi4（`out_review_neutral_bind`）+ qwen38（`out_review_neutral_third_bind`）三读者判，
   走**同一个 ≥2/3 闸**，再叠加**逐绑定 ≤20% 反对闸**；回收者改标 `implied_negative_v2`。
   （audit 批次里这些行 readers=None 只是 assemble 没去 neutral_bind 取票，不是没审。）
6. `organ_backed_negative`：器官段「Normal」背书的 finding 阴性，单独打标。
7. **发布组装**：在保留集上剔除 `organ_backed_negative` finding、移除 4 个退化 organ、
   下采样 organ `no_abnormality`（保留集里占 81%，平凡基线过高），并纳入 newq ⇒ 280,111。
8. **人工抽检**：见 §6。

---

## 5. LLM reader consistency threshold（需求 5）

- **主闸 V2：3 读者 ≥2 票 Support**（两票制，非相关系数阈值）。
- **implied_negative_v2 附加闸：逐绑定反对率 ≤20%**。
- 平票由第三读者 Qwen3.8-27B 裁决。

**读者同意率 / 分歧 / 第三读者裁决（实测账）**

IID（全族，读者核验阶段；源 `out_review/third_reader_results.md`、`cross_reader*.md`）：
- 两基座读者都认规则、无需裁决：**333,202 items**。
- finding 族同意率 **96.0%**（82,836 条双判文本里 79,494 都认）；读者间一致 **97.8%，κ=0.955**。
  organ 族较难：都认 73,588/110,375=66.7%，读者间一致 86.6%，κ=0.826。
- 两者分歧（恰一票 S）→ 第三读者：**26,932 items**；第三读者**救回/保留 6,790（25.2%）、丢弃 20,142（74.8%）**。
  - finding 单族：分歧 1,854 条 → 留 1,564 / 弃 290。
  - organ 单族：分歧 13,082 条 → 留 2,941（items 3,029）/ 弃其余。

OOD（the external OOD cohort，613 条审计；源 `out_ood_v3_audit/merged.jsonl`）：
- 两读者都认规则 **567=92.5%**；都反对 26=4.2%；分歧 **19=3.1%**；读者间一致 96.7%。
- 第三读者对 19 分歧：**留 9 / 弃 10**。

---

## 6. Human audit sample size（需求 6）

- **IID：400 条**（`out_iid_audit400/`，seed=20261115，从清洗保留集 keep=1 随机抽）：
  **388 OK / 12 WEAK / 0 WRONG**（错误率 0%，WEAK 3.0%）。
  12 WEAK = 7 organ 选项集覆盖不足 + 3 证据混合/含糊 + 2 读者边界分歧；无标签错误。
  （注：抽检对象是清洗保留集；发布文件在其上再做 organ 下采样/去退化器官并加 newq。）
- **OOD：200 条**（记录在案：4 WRONG / 8 WEAK）。
- 判读标准：OK=证据支持 gold 或 gold 为最佳 MC 选项且读者一致；WEAK=边界/混合/选项覆盖；WRONG=证据与 gold 矛盾。

---

## 7. 丢弃问题比例（需求 7）

**清洗闸（label verification 阶段）**：出厂 733,452 → keep=1 395,971 / keep=0 337,481，**丢弃率 46.0%**。

| keep=0 原因 | n |
|---|---|
| implied_boilerplate_only（仅模板套话） | 79,551 |
| not_auditable（不可审计） | 73,519 |
| implied_binding_oppose_above_threshold（绑定反对 >20%） | 53,775 |
| implied_opposed | 48,126 |
| readers_not_agree（读者不一致） | 44,742 |
| implied_no_valid_vote | 27,929 |
| implied_direction_wrong | 5,278 |
| implied_gate_v2_requires_two_of_three（不足两票） | 4,200 |
| negated_hedged_excluded | 361 |

**发布组装再过滤**：keep=1 395,971 → 发布 280,111（剔 organ_backed_negative finding 39,919、
移除 4 退化 organ、下采样 organ no_abnormality；另纳入 newq 26,553 及新增项）。

---

## 8. OOD 与 IID 的分布差异（需求 10）

| 维度 | IID（the source cohort，实际文件） | OOD（the external OOD cohort） |
|---|---|---|
| 来源 | the source cohort 腹部 CT，报告衍生 | the external OOD cohort 外部（`the external OOD cohort/imagesTr/ood_*.nii.gz`） |
| split | train/val/test | **仅 test**（prediction-only，不进拟合/不驱动选模） |
| study 数 | 25,270 | 421 |
| 题量 | 280,111 | 2,807（冻结 test，版本敏感） |
| 答案类型 | noul 187,189 / choice 79,711 / score 13,211 | noul 1,947 / choice 801 / score 59 |
| finding 病种 | 30 | 24 |
| organ | 8 | 8 组（含合并组 lymph_spleen_prostate_pancreas、peritoneal；另 adrenal 1 条） |
| 年龄/性别 | 有 | 常「age not available」 |

**关键差异**：
- OOD **缺 6 个 IID 病种**：atelectasis / pleural_effusion / coronary_calcification /
  aortic_valve_calcification / cardiomegaly（5 个胸腔下部）+ anasarca —— the external OOD cohort 偏腹部。
- **患病率排序不同**：IID 以常见/incidental 为主（gallstones、atelectasis、renal_hypodensities 居前）；
  OOD 富集病理（lymphadenopathy 300 / ascites 179 / splenomegaly 148 / renal_cyst 146 / biliary_ductal_dilation 131）。
- OOD 用合并器官组 + 整图回退（lymph/peritoneal/prostate 无类目→整图）；IID 逐器官绑定。
- **泄漏约束**：OOD gold 不进拟合、OOD 读出不驱动选模，仅预测。

---

### 仍需钉版（勿擅自写死）
1. OOD 总数：当前冻结 test=2,807；历史 2577/2823 未钉。
2. caption「24 findings」与 30 finding 病种 / 21 scored 的口径对齐。