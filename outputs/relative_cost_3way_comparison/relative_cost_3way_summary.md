# Relative Cost 3-Way Comparison

一次 feature extraction 同时输出 `geo`、`target_barrier` (`tbar`) 与 `symmetric_barrier` (`sbar`) 三套 transport risk；下面分类结果均为 XGB、同一 split、按 AUC grid-search。

## Best Cost By Family

### Qwen2.5-VL-7B
| Family | Best | AUC | F1 | Geo AUC | Delta vs geo | All AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | tbar | 0.895 | 0.907 | 0.892 | 0.003 | geo=0.892 / tbar=0.895 / sbar=0.870 |
| Visual risk cap085 | geo | 0.898 | 0.905 | 0.898 | 0.000 | geo=0.898 / tbar=0.866 / sbar=0.860 |
| Visual+prompt risk | geo | 0.926 | 0.922 | 0.926 | 0.000 | geo=0.926 / tbar=0.922 / sbar=0.881 |
| Visual+prompt risk cap085 | tbar | 0.924 | 0.917 | 0.916 | 0.008 | geo=0.916 / tbar=0.924 / sbar=0.888 |
| Visual risk + visual cosine | tbar | 0.939 | 0.939 | 0.933 | 0.006 | geo=0.933 / tbar=0.939 / sbar=0.917 |
| Visual risk cap085 + visual cosine cap085 | geo | 0.925 | 0.927 | 0.925 | 0.000 | geo=0.925 / tbar=0.916 / sbar=0.912 |
| Visual+prompt risk + VP cosine | tbar | 0.946 | 0.933 | 0.946 | 0.000 | geo=0.946 / tbar=0.946 / sbar=0.927 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.947 | 0.929 | 0.940 | 0.007 | geo=0.940 / tbar=0.947 / sbar=0.922 |

### InternVL2.5-8B
| Family | Best | AUC | F1 | Geo AUC | Delta vs geo | All AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | geo | 0.888 | 0.933 | 0.888 | 0.000 | geo=0.888 / tbar=0.882 / sbar=0.858 |
| Visual risk cap085 | tbar | 0.877 | 0.943 | 0.870 | 0.007 | geo=0.870 / tbar=0.877 / sbar=0.846 |
| Visual+prompt risk | tbar | 0.943 | 0.951 | 0.931 | 0.012 | geo=0.931 / tbar=0.943 / sbar=0.920 |
| Visual+prompt risk cap085 | tbar | 0.928 | 0.948 | 0.921 | 0.007 | geo=0.921 / tbar=0.928 / sbar=0.926 |
| Visual risk + visual cosine | geo | 0.959 | 0.957 | 0.959 | 0.000 | geo=0.959 / tbar=0.947 / sbar=0.941 |
| Visual risk cap085 + visual cosine cap085 | geo | 0.953 | 0.954 | 0.953 | 0.000 | geo=0.953 / tbar=0.951 / sbar=0.946 |
| Visual+prompt risk + VP cosine | tbar | 0.960 | 0.955 | 0.959 | 0.001 | geo=0.959 / tbar=0.960 / sbar=0.946 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.953 | 0.952 | 0.947 | 0.006 | geo=0.947 / tbar=0.953 / sbar=0.940 |

### LLaVA-1.5-7B
| Family | Best | AUC | F1 | Geo AUC | Delta vs geo | All AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual risk | geo | 0.818 | 0.746 | 0.818 | 0.000 | geo=0.818 / tbar=0.815 / sbar=0.792 |
| Visual risk cap085 | tbar | 0.819 | 0.741 | 0.771 | 0.049 | geo=0.771 / tbar=0.819 / sbar=0.806 |
| Visual+prompt risk | sbar | 0.795 | 0.672 | 0.776 | 0.019 | geo=0.776 / tbar=0.790 / sbar=0.795 |
| Visual+prompt risk cap085 | tbar | 0.799 | 0.755 | 0.795 | 0.004 | geo=0.795 / tbar=0.799 / sbar=0.789 |
| Visual risk + visual cosine | tbar | 0.853 | 0.777 | 0.830 | 0.023 | geo=0.830 / tbar=0.853 / sbar=0.841 |
| Visual risk cap085 + visual cosine cap085 | sbar | 0.819 | 0.721 | 0.800 | 0.019 | geo=0.800 / tbar=0.817 / sbar=0.819 |
| Visual+prompt risk + VP cosine | geo | 0.842 | 0.741 | 0.842 | 0.000 | geo=0.842 / tbar=0.831 / sbar=0.835 |
| Visual+prompt risk cap085 + VP cosine cap085 | tbar | 0.832 | 0.769 | 0.823 | 0.009 | geo=0.823 / tbar=0.832 / sbar=0.824 |

## Layerwise Diff Summary

### Qwen2.5-VL-7B
| Group | Cost | Hall mean | Non mean | Diff avg | Peak layer | Peak diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_raw | geo | 0.302 | 0.282 | 0.021 | 27 | -0.096 |
| visual_raw | tbar | 0.320 | 0.295 | 0.024 | 27 | -0.100 |
| visual_raw | sbar | 0.415 | 0.382 | 0.034 | 19 | 0.127 |
| visual_cap085 | geo | 0.280 | 0.265 | 0.015 | 27 | -0.104 |
| visual_cap085 | tbar | 0.296 | 0.278 | 0.018 | 27 | -0.109 |
| visual_cap085 | sbar | 0.385 | 0.360 | 0.026 | 19 | 0.121 |
| vp_raw | geo | 0.349 | 0.326 | 0.023 | 27 | -0.127 |
| vp_raw | tbar | 0.367 | 0.342 | 0.025 | 27 | -0.127 |
| vp_raw | sbar | 0.566 | 0.522 | 0.044 | 27 | -0.462 |
| vp_cap085 | geo | 0.348 | 0.325 | 0.023 | 27 | -0.127 |
| vp_cap085 | tbar | 0.367 | 0.343 | 0.025 | 27 | -0.130 |
| vp_cap085 | sbar | 0.552 | 0.513 | 0.039 | 27 | -0.470 |

### InternVL2.5-8B
| Group | Cost | Hall mean | Non mean | Diff avg | Peak layer | Peak diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_raw | geo | 0.293 | 0.298 | -0.005 | 8 | -0.056 |
| visual_raw | tbar | 0.304 | 0.309 | -0.005 | 8 | -0.061 |
| visual_raw | sbar | 0.362 | 0.358 | 0.003 | 8 | -0.050 |
| visual_cap085 | geo | 0.279 | 0.284 | -0.004 | 8 | -0.050 |
| visual_cap085 | tbar | 0.290 | 0.295 | -0.005 | 8 | -0.054 |
| visual_cap085 | sbar | 0.352 | 0.350 | 0.002 | 8 | -0.045 |
| vp_raw | geo | 0.513 | 0.481 | 0.032 | 8 | 0.089 |
| vp_raw | tbar | 0.561 | 0.523 | 0.038 | 1 | 0.113 |
| vp_raw | sbar | 0.700 | 0.654 | 0.046 | 31 | 0.189 |
| vp_cap085 | geo | 0.529 | 0.498 | 0.031 | 8 | 0.080 |
| vp_cap085 | tbar | 0.576 | 0.540 | 0.037 | 1 | 0.112 |
| vp_cap085 | sbar | 0.727 | 0.681 | 0.046 | 31 | 0.171 |

### LLaVA-1.5-7B
| Group | Cost | Hall mean | Non mean | Diff avg | Peak layer | Peak diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| visual_raw | geo | 0.496 | 0.472 | 0.024 | 22 | 0.087 |
| visual_raw | tbar | 0.525 | 0.497 | 0.029 | 22 | 0.092 |
| visual_raw | sbar | 0.696 | 0.650 | 0.046 | 25 | 0.140 |
| visual_cap085 | geo | 0.471 | 0.456 | 0.015 | 0 | -0.077 |
| visual_cap085 | tbar | 0.500 | 0.481 | 0.019 | 0 | -0.078 |
| visual_cap085 | sbar | 0.668 | 0.640 | 0.028 | 25 | 0.115 |
| vp_raw | geo | 0.626 | 0.655 | -0.028 | 31 | -0.047 |
| vp_raw | tbar | 0.659 | 0.691 | -0.031 | 1 | -0.070 |
| vp_raw | sbar | 0.874 | 0.895 | -0.021 | 8 | -0.107 |
| vp_cap085 | geo | 0.627 | 0.663 | -0.036 | 31 | -0.053 |
| vp_cap085 | tbar | 0.660 | 0.699 | -0.038 | 31 | -0.065 |
| vp_cap085 | sbar | 0.886 | 0.926 | -0.040 | 8 | -0.151 |

## Quick Read

- Qwen: `geo` 是整体最稳的默认 cost；`tbar` 只在少数 visual-only raw/combo 上小幅超过 `geo`，`sbar` 通常不占优。
- InternVL: visual+prompt risk-only 的 raw AUC 更偏向 `tbar`，但加 cosine 后 `geo`/`tbar` 非常接近；`sbar` 基本弱于前两者。
- LLaVA: 单独 visual risk 里 `geo/tbar` 接近，最高 raw/capped risk 都在约 0.82；加 cosine 后 `tbar` 在 visual branch 最强，AUC 约 0.853；visual+prompt combo 则 `geo` 略高，AUC 约 0.842。
- 从曲线看，barrier cost 会抬高 risk 绝对值，`sbar` 抬得最明显；但抬高均值不等于更好的分类，三模型上 `sbar` 都不适合作为默认。
