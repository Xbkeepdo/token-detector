# Relative Cost 3-Way: MLP vs XGB

同一批 3-way features、同一 split；每个 family 内比较 `geo/tbar/sbar` 三个 cost 的最佳 XGB 与最佳 MLP。MLP 训练中出现的 `ConvergenceWarning` 表示部分网格配置 500 iter 未完全收敛，但结果已按验证 AUC 选最优配置。

## Overall Best

| Model | Clf | Family | Cost | AUC | F1 | Prec | Rec | Feature Set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | xgb | vp_combo_cap085 | tbar | 0.947 | 0.929 | 0.912 | 0.948 | risk_visual_prompt_relative_vll_cost_tbar_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 |
| Qwen2.5-VL-7B | mlp | vp_combo_raw | sbar | 0.940 | 0.914 | 0.861 | 0.974 | risk_visual_prompt_relative_vll_cost_sbar+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll |
| InternVL2.5-8B | xgb | vp_combo_raw | tbar | 0.960 | 0.955 | 0.946 | 0.965 | risk_visual_prompt_relative_vll_cost_tbar+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll |
| InternVL2.5-8B | mlp | visual_combo_raw | sbar | 0.950 | 0.951 | 0.943 | 0.958 | risk_relative_vll_cost_sbar+target_visual_hidden_cosine_relative_vll |
| LLaVA-1.5-7B | xgb | visual_combo_raw | tbar | 0.853 | 0.777 | 0.761 | 0.794 | risk_relative_vll_cost_tbar+target_visual_hidden_cosine_relative_vll |
| LLaVA-1.5-7B | mlp | vp_combo_cap085 | sbar | 0.852 | 0.769 | 0.682 | 0.882 | risk_visual_prompt_relative_vll_cost_sbar_capped_topmass_085+target_visual_prompt_hidden_cosine_visual_prompt_relative_vll_capped_topmass_085 |

## Best By Family

### Qwen2.5-VL-7B
| Family | XGB Cost | XGB AUC | MLP Cost | MLP AUC | MLP-XGB | MLP F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | tbar | 0.895 | tbar | 0.897 | 0.001 | 0.895 |
| Visual risk cap085 | geo | 0.898 | geo | 0.911 | 0.013 | 0.889 |
| Visual+prompt risk | geo | 0.926 | geo | 0.887 | -0.040 | 0.901 |
| Visual+prompt risk cap085 | tbar | 0.924 | tbar | 0.889 | -0.034 | 0.905 |
| Visual risk + visual cosine | tbar | 0.939 | tbar | 0.932 | -0.007 | 0.914 |
| Visual risk cap085 + visual cosine cap085 | geo | 0.925 | tbar | 0.926 | 0.001 | 0.912 |
| Visual+prompt risk + VP cosine | tbar | 0.946 | sbar | 0.940 | -0.006 | 0.914 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.947 | sbar | 0.935 | -0.012 | 0.912 |

### InternVL2.5-8B
| Family | XGB Cost | XGB AUC | MLP Cost | MLP AUC | MLP-XGB | MLP F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | geo | 0.888 | sbar | 0.885 | -0.003 | 0.933 |
| Visual risk cap085 | tbar | 0.877 | tbar | 0.881 | 0.004 | 0.940 |
| Visual+prompt risk | tbar | 0.943 | tbar | 0.925 | -0.018 | 0.944 |
| Visual+prompt risk cap085 | tbar | 0.928 | sbar | 0.919 | -0.010 | 0.941 |
| Visual risk + visual cosine | geo | 0.959 | sbar | 0.950 | -0.009 | 0.951 |
| Visual risk cap085 + visual cosine cap085 | geo | 0.953 | tbar | 0.934 | -0.019 | 0.944 |
| Visual+prompt risk + VP cosine | tbar | 0.960 | sbar | 0.940 | -0.020 | 0.955 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.953 | geo | 0.941 | -0.012 | 0.953 |

### LLaVA-1.5-7B
| Family | XGB Cost | XGB AUC | MLP Cost | MLP AUC | MLP-XGB | MLP F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | geo | 0.818 | sbar | 0.786 | -0.032 | 0.707 |
| Visual risk cap085 | tbar | 0.819 | sbar | 0.784 | -0.036 | 0.734 |
| Visual+prompt risk | sbar | 0.795 | geo | 0.799 | 0.004 | 0.696 |
| Visual+prompt risk cap085 | tbar | 0.799 | tbar | 0.788 | -0.011 | 0.708 |
| Visual risk + visual cosine | tbar | 0.853 | geo | 0.846 | -0.006 | 0.781 |
| Visual risk cap085 + visual cosine cap085 | sbar | 0.819 | geo | 0.846 | 0.027 | 0.750 |
| Visual+prompt risk + VP cosine | geo | 0.842 | sbar | 0.838 | -0.004 | 0.709 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.832 | sbar | 0.852 | 0.020 | 0.769 |

## Quick Read

- Qwen: MLP 对单独 visual risk 有收益，尤其 `risk_relative_vll_cost_geo_capped_topmass_085` AUC 0.911；但组合项最高 MLP 约 0.940，仍低于 XGB 最高 0.947。
- InternVL: MLP 没有超过 XGB 总体最强，最高约 0.950（visual `sbar+cosine`），低于 XGB 最高约 0.960。
- LLaVA: MLP 最高约 0.852，几乎追平 XGB 最高 0.853；最好项从 XGB 的 visual `tbar+cosine` 转到 MLP 的 VP capped `sbar+cosine`，但优势不成立。
- 三模型合看：MLP 可以作为 sanity check/补充，但当前 cost 3-way 主结论仍应以 XGB 为主；`sbar` 偶尔在 MLP 组合里变强，但没有稳定超过 `geo/tbar`。
