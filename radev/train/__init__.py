"""RADAR(冻结视觉编码器) + projector + 小型 LLM 的 Jev 形式训练。

分工（✅ = 已有实现；其余**还没写**，这张表不是模块索引的替代品）：
    geometry      ✅ RADAR 输入几何的可行性扫描（先于 GPU 工作）
    volume        ✅ 体数据预处理，逐字复用上游口径（HU clip / resample / pad / 滑窗）
    radar_encoder ✅ VisionBranch 移植层 + 滑窗读出，产出每器官一个 256-d 嵌入（a17 拒绝）
    vendor/       ✅ RADAR_inference/dynamic_network_architectures 整包移植（保留上游许可头）
    parity_demo   ✅ 步骤 3 对拍：demo 一例逐位复现自带 CSV（20 概率 + sag_voxels）
    cache         ✅ stage 1：全库器官嵌入落盘（plan/run，a15 编码器哈希断言）；
                  真跑入口 run_radar_cache.sh（GO=1 门控）
    projector     mlp2x_gelu 256 -> D_llm
    pack          Jev 记录 -> (input_ids, labels, candidate_ids)，答案模板保证合法 Jev 应答
    dataset       study 级 shuffle、题级 batch 的数据集（一序列一题，避免跨题标签捷径）
    model         三段组装 + 受限 softmax 读出
    floor         两级地板：state-only（人口学）与 radar-cosine（编码器能力）
    metrics       逐题 balanced accuracy / AUROC / QWK / level MAE + 地板显著性判定
    loop          训练循环（bf16 + 梯度累积 + 检查点续跑 + --limit dry run）

纪律（都是踩过的坑，不是装饰）：
    1. 加载的 checkpoint 必须是 checkpoint_radar_pretrain.pth。
       用 checkpoint_radar_plus_finetuned.pth 会让 RADAR 的概率从「第三方佐证」
       退化成「在 the source cohort 上训过的拟合器」，② 级地板当场失效。见 a17。
    2. --rules 无默认值。判据真值只在这几处：out_review/rule_decision.md（四条极性规则）、
       out_review_neutral_full/implied_negative_allow.tsv（implied 逐格）、
       analysis/gate.py（逐 item 谓词与档位）。代码里不抄第二份数字 ——
       本文件上一版写死的「4 坏 / 8 放行 / 18 中间」是 14B 旧配对的作废值。
    3. 不许改上游预处理来"救"被 [-300,400] clip 饱和掉的钙化/结石信号 —— in_channels=1
       加 BatchNorm3d，改了就不是 RADAR。改为标记 + 计数。
    4. ③ 级不显著超过 ① ② 两级地板的题，拒绝报数。
"""
