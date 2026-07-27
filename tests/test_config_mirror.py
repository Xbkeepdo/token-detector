"""Keep unified and fj01 YAML behavior identical apart from host paths."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from unittest import TestCase

from utils.config_utils import load_config


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT_PATH_FIELDS = (
    ("dataset", "coco_root"),
    ("dataset", "annotation_file"),
    ("dataset", "captions_file"),
    ("feature_extraction", "baseline", "halloc", "clip_model"),
    ("feature_extraction", "baseline", "halloc", "visualbert_model"),
    ("models", "llava_1_5_7b", "hf_name"),
    ("models", "llava_next_8b", "hf_name"),
    ("models", "llava_onevision_1_5_8b", "hf_name"),
    ("models", "llava_onevision_1_5_8b_instruct", "hf_name"),
    ("models", "internvl_2_5_8b", "hf_name"),
    ("models", "qwen2_5_vl_7b", "hf_name"),
    ("models", "qwen3_vl_8b", "hf_name"),
    ("qa_benchmarks", "pope_dir"),
    ("qa_benchmarks", "coco_image_dir"),
    ("qa_benchmarks", "clevr_root"),
    ("qa_benchmarks", "amber_root"),
    ("qa_benchmarks", "prepared_root"),
    ("qa_benchmarks", "output_root"),
)


def _without_environment_paths(config: dict) -> dict:
    result = deepcopy(config)
    for path in ENVIRONMENT_PATH_FIELDS:
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        assert path[-1] in parent, f"Missing configured path: {'.'.join(path)}"
        parent.pop(path[-1])
    return result


class ConfigMirrorTests(TestCase):
    def test_unified_and_fj01_have_identical_runtime_semantics(self) -> None:
        unified = load_config(str(ROOT / "configs" / "model_configs_unified.yaml"))
        fj01 = load_config(
            str(ROOT / "configs" / "model_configs_server_fj01.yaml")
        )
        self.assertEqual(
            _without_environment_paths(unified),
            _without_environment_paths(fj01),
        )
