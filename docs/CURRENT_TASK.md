# Current Task

## Goal
比较 Qwen2.5-VL-7B 与 InternVL2.5-8B 在 DGST-T `support_scope=visual` 时，幻觉/非幻觉 object token 的逐层 transport risk 曲线。

## Last status
- 已新增 `configs/model_configs_visualonly.yaml`，只在 Qwen/InternVL 模型配置中设置 `dgst_t_support_scope: "visual"`，并修正 COCO 数据路径为 `/home/apulis-dev/userdata/DGST/token-grounding-detector/data/coco`。
- 用户要求将 visual-only 的 DGST-T `cost_mode` 改为 `decomposed` 后重跑；已确认并将 `configs/model_configs_visualonly.yaml` 设置为 `cost_mode: "decomposed"`。
- 已完成 InternVL COCO500 visual-only 特征抽取：`outputs/internvl_2_5_8b/COCO500-visualonly/features.pkl`，3715 行，hall=3064，true=651，support_size=256。
- 已完成 Qwen COCO500 visual-only 特征抽取：`outputs/qwen2_5_vl_7b/COCO500-visualonly/features.pkl`，2190 行，hall=1636，true=554。Qwen visual token 数随样本变化，首层 support_size 范围约 63-529，均值约 348.96。
- 已生成逐层 risk 曲线与 CSV：
  - `outputs/qwen2_5_vl_7b/COCO500-visualonly/results/qwen2_5_vl_7b_visualonly_decomposed_risk_layerwise_by_label.{png,pdf,csv}`
  - `outputs/internvl_2_5_8b/COCO500-visualonly/results/internvl_2_5_8b_visualonly_decomposed_risk_layerwise_by_label.{png,pdf,csv}`
  - `outputs/visualonly_support_risk_comparison/qwen_internvl_visualonly_decomposed_risk_layerwise_by_label.{png,pdf}`
- decomposed 初步结果：Qwen visual-only 下 hallucination 与 non-hallucination 的平均 risk 曲线仍高度重合，diff_avg 约 0.0011，最大绝对差在第 12 层约 -0.0889；InternVL visual-only 下早期层差异更明显，diff_avg 约 0.0151，最大绝对差在第 3 层约 0.169。
- 与 direct 版本相比，decomposed 主要使 risk 绝对值整体下移，曲线形状和类别差异趋势基本一致。

## Files changed recently
- 新增：`configs/model_configs_visualonly.yaml`
- 新增输出目录：`outputs/internvl_2_5_8b/COCO500-visualonly/`
- 新增输出目录：`outputs/qwen2_5_vl_7b/COCO500-visualonly/`
- 新增输出目录：`outputs/visualonly_support_risk_comparison/`
- 备份错误中间结果：`outputs/qwen2_5_vl_7b/COCO500-visualonly/bad_internvl_labels_20260630_194840/`

## Commands run
- `CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model internvl_2_5_8b --config configs/model_configs_visualonly.yaml --output-dir outputs/internvl_2_5_8b/COCO500-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
- `CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 /opt/conda/private/envs/vicr/bin/python scripts/extract_features.py --model qwen2_5_vl_7b --config configs/model_configs_visualonly.yaml --output-dir outputs/qwen2_5_vl_7b/COCO500-visualonly --device cuda:0 --feature-devices cuda:0 cuda:1 --resume`
- 使用 `/opt/conda/private/envs/vicr/bin/python` 读取 `features.pkl` 并生成逐层 risk 的 png/pdf/csv。
- 2026-06-30 再次运行上述两条 `extract_features.py` 命令，用 `cost_mode: "decomposed"` 重算 visual-only 特征和图。

## Known issues
- 第一次 Qwen visual-only 抽取误用了 InternVL 的 `labeling.json/generations.json`，产物已移动到 `outputs/qwen2_5_vl_7b/COCO500-visualonly/bad_internvl_labels_20260630_194840/`，不要用于分析。
- 默认 `/opt/conda/bin/python` 缺少 torch/transformers/scipy/sklearn/openai/pyyaml；本次运行使用的是 `/opt/conda/private/envs/vicr/bin/python`。
- `run.sh` 中仍有疑似 API key 示例字符串，建议后续清理并确认是否需要轮换。

## Next steps
- 如需进一步量化 visual-only 是否改善检测，可在两个 visual-only 输出目录上运行 `scripts/train_feature_sets.py` 或 torch probe。
- 如需和 visual+prompt 直接对比，可把原 `COCO500/results/*risk_layerwise_by_label.csv` 与本次 visual-only CSV 做同图差值分析。
