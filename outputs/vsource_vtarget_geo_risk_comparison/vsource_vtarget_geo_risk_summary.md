# V Source / V Target / Geo Risk Curves

本轮三模型都使用 `dgst_t_support_scope: visual`，因此 source distribution 和 relative-VLL target candidate 都限制在 visual tokens；cost mode 为 `geo`。

| Model | n hall | n non | Hall mean | Non mean | Diff avg | Peak layer | Peak diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 1636 | 554 | 0.249 | 0.228 | 0.021 | 19 | 0.080 |
| InternVL2.5-8B | 3064 | 651 | 0.255 | 0.247 | 0.008 | 2 | 0.043 |
| LLaVA-1.5-7B | 691 | 757 | 0.478 | 0.451 | 0.027 | 22 | 0.087 |

## Files

- `vsource_vtarget_geo_risk_3models_by_label.{png,pdf}`: 三模型合并图。
- `{model}_vsource_vtarget_geo_risk_by_label.{png,pdf,csv}`: 单模型曲线和逐层数值。
