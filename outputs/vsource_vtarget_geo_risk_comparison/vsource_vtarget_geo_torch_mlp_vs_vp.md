# V/V Geo Torch MLP vs VP/VP Geo
Classifier: torch MLP/probe. Positive class: hallucination. Cost: geo.
`V/V` means source and target support are both visual tokens via `configs/model_configs_visualonly_relativevll_cost_geo.yaml`. `VP/VP` means source and target support are both visual+prompt tokens, using the `risk_visual_prompt_relative_vll_cost_geo*` fields from additive3 outputs.
AUPR is average precision on the same held-out test split, computed from the saved torch probe checkpoints.
## Best AUC by Model
| Model | Best V/V feature | V/V AUC | V/V AUPR | V/V F1 | Best VP/VP feature | VP/VP AUC | VP/VP AUPR | VP/VP F1 | Better | ΔAUC V/V-VP | ΔAUPR V/V-VP |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| Qwen2.5-VL-7B | risk + cosine / cap085 | 0.952 | 0.984 | 0.929 | risk + cosine / raw | 0.923 | 0.966 | 0.920 | V/V | 0.029 | 0.018 |
| InternVL2.5-8B | risk + cosine / raw | 0.942 | 0.982 | 0.944 | risk + cosine / raw | 0.952 | 0.987 | 0.965 | VP/VP | -0.010 | -0.005 |
| LLaVA-1.5-7B | risk + cosine / raw | 0.827 | 0.782 | 0.713 | risk + cosine / raw | 0.844 | 0.844 | 0.769 | VP/VP | -0.017 | -0.062 |

## Best AUPR by Model
| Model | Best V/V feature | V/V AUPR | V/V AUC | Best VP/VP feature | VP/VP AUPR | VP/VP AUC | Better | ΔAUPR V/V-VP |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| Qwen2.5-VL-7B | risk + cosine / cap085 | 0.984 | 0.952 | risk + cosine / raw | 0.966 | 0.923 | V/V | 0.018 |
| InternVL2.5-8B | risk + cosine / raw | 0.982 | 0.942 | risk only / cap085 | 0.987 | 0.941 | VP/VP | -0.005 |
| LLaVA-1.5-7B | risk + cosine / cap085 | 0.782 | 0.817 | risk + cosine / raw | 0.844 | 0.844 | VP/VP | -0.062 |

## Matched Feature Comparison
| Model | Variant | V/V AUC | VP/VP AUC | ΔAUC | V/V AUPR | VP/VP AUPR | ΔAUPR | V/V F1 | VP/VP F1 | ΔF1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | risk only / raw | 0.909 | 0.878 | 0.031 | 0.967 | 0.946 | 0.021 | 0.898 | 0.902 | -0.004 |
| Qwen2.5-VL-7B | risk only / cap085 | 0.894 | 0.862 | 0.032 | 0.964 | 0.937 | 0.027 | 0.880 | 0.903 | -0.023 |
| Qwen2.5-VL-7B | risk + cosine / raw | 0.945 | 0.923 | 0.022 | 0.980 | 0.966 | 0.014 | 0.932 | 0.920 | 0.012 |
| Qwen2.5-VL-7B | risk + cosine / cap085 | 0.952 | 0.919 | 0.033 | 0.984 | 0.963 | 0.020 | 0.929 | 0.920 | 0.008 |
| InternVL2.5-8B | risk only / raw | 0.840 | 0.937 | -0.097 | 0.958 | 0.987 | -0.029 | 0.925 | 0.960 | -0.035 |
| InternVL2.5-8B | risk only / cap085 | 0.848 | 0.941 | -0.093 | 0.961 | 0.987 | -0.026 | 0.928 | 0.961 | -0.033 |
| InternVL2.5-8B | risk + cosine / raw | 0.942 | 0.952 | -0.010 | 0.982 | 0.987 | -0.005 | 0.944 | 0.965 | -0.021 |
| InternVL2.5-8B | risk + cosine / cap085 | 0.936 | 0.939 | -0.003 | 0.981 | 0.983 | -0.002 | 0.948 | 0.965 | -0.018 |
| LLaVA-1.5-7B | risk only / raw | 0.762 | 0.781 | -0.019 | 0.718 | 0.759 | -0.041 | 0.685 | 0.701 | -0.016 |
| LLaVA-1.5-7B | risk only / cap085 | 0.774 | 0.763 | 0.010 | 0.733 | 0.711 | 0.022 | 0.690 | 0.711 | -0.021 |
| LLaVA-1.5-7B | risk + cosine / raw | 0.827 | 0.844 | -0.017 | 0.782 | 0.844 | -0.062 | 0.713 | 0.769 | -0.056 |
| LLaVA-1.5-7B | risk + cosine / cap085 | 0.817 | 0.830 | -0.013 | 0.782 | 0.814 | -0.031 | 0.750 | 0.740 | 0.010 |

## Full Metrics
| Model | Mode | Variant | Precision | Recall | F1 | Accuracy | AUC | AUPR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | V/V | risk only / raw | 0.901 | 0.895 | 0.898 | 0.850 | 0.909 | 0.967 |
| Qwen2.5-VL-7B | VP/VP | risk only / raw | 0.872 | 0.935 | 0.902 | 0.850 | 0.878 | 0.946 |
| Qwen2.5-VL-7B | V/V | risk only / cap085 | 0.853 | 0.908 | 0.880 | 0.816 | 0.894 | 0.964 |
| Qwen2.5-VL-7B | VP/VP | risk only / cap085 | 0.867 | 0.941 | 0.903 | 0.850 | 0.862 | 0.937 |
| Qwen2.5-VL-7B | V/V | risk + cosine / raw | 0.923 | 0.941 | 0.932 | 0.899 | 0.945 | 0.980 |
| Qwen2.5-VL-7B | VP/VP | risk + cosine / raw | 0.905 | 0.935 | 0.920 | 0.879 | 0.923 | 0.966 |
| Qwen2.5-VL-7B | V/V | risk + cosine / cap085 | 0.923 | 0.935 | 0.929 | 0.894 | 0.952 | 0.984 |
| Qwen2.5-VL-7B | VP/VP | risk + cosine / cap085 | 0.900 | 0.941 | 0.920 | 0.879 | 0.919 | 0.963 |
| InternVL2.5-8B | V/V | risk only / raw | 0.883 | 0.971 | 0.925 | 0.868 | 0.840 | 0.958 |
| InternVL2.5-8B | VP/VP | risk only / raw | 0.947 | 0.974 | 0.960 | 0.932 | 0.937 | 0.987 |
| InternVL2.5-8B | V/V | risk only / cap085 | 0.888 | 0.971 | 0.928 | 0.873 | 0.848 | 0.961 |
| InternVL2.5-8B | VP/VP | risk only / cap085 | 0.941 | 0.981 | 0.961 | 0.932 | 0.941 | 0.987 |
| InternVL2.5-8B | V/V | risk + cosine / raw | 0.934 | 0.955 | 0.944 | 0.905 | 0.942 | 0.982 |
| InternVL2.5-8B | VP/VP | risk + cosine / raw | 0.953 | 0.977 | 0.965 | 0.941 | 0.952 | 0.987 |
| InternVL2.5-8B | V/V | risk + cosine / cap085 | 0.937 | 0.958 | 0.948 | 0.911 | 0.936 | 0.981 |
| InternVL2.5-8B | VP/VP | risk + cosine / cap085 | 0.950 | 0.981 | 0.965 | 0.941 | 0.939 | 0.983 |
| LLaVA-1.5-7B | V/V | risk only / raw | 0.653 | 0.721 | 0.685 | 0.685 | 0.762 | 0.718 |
| LLaVA-1.5-7B | VP/VP | risk only / raw | 0.712 | 0.691 | 0.701 | 0.720 | 0.781 | 0.759 |
| LLaVA-1.5-7B | V/V | risk only / cap085 | 0.649 | 0.735 | 0.690 | 0.685 | 0.774 | 0.733 |
| LLaVA-1.5-7B | VP/VP | risk only / cap085 | 0.643 | 0.794 | 0.711 | 0.692 | 0.763 | 0.711 |
| LLaVA-1.5-7B | V/V | risk + cosine / raw | 0.680 | 0.750 | 0.713 | 0.713 | 0.827 | 0.782 |
| LLaVA-1.5-7B | VP/VP | risk + cosine / raw | 0.733 | 0.809 | 0.769 | 0.769 | 0.844 | 0.844 |
| LLaVA-1.5-7B | V/V | risk + cosine / cap085 | 0.711 | 0.794 | 0.750 | 0.748 | 0.817 | 0.782 |
| LLaVA-1.5-7B | VP/VP | risk + cosine / cap085 | 0.692 | 0.794 | 0.740 | 0.734 | 0.830 | 0.814 |

## Files
- Full metrics CSV: `outputs/vsource_vtarget_geo_risk_comparison/vsource_vtarget_geo_torch_mlp_vs_vp_all.csv`
- Delta CSV: `outputs/vsource_vtarget_geo_risk_comparison/vsource_vtarget_geo_torch_mlp_vs_vp_delta.csv`
- V/V result JSONs: `outputs/*/COCO500-visualonly-relativevll-cost-geo/results/*_selected_feature_sets.json`
- VP/VP result JSONs: `outputs/*/COCO500-visualprompt-relativevll-cost-additive3/results/*_selected_feature_sets.json`
