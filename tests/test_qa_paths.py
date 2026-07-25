from pathlib import Path
import unittest

from utils.qa_paths import (
    DEFAULT_QA_OUTPUT,
    generations_path_for_benchmark_dir,
    locate_qa_generations,
    resolve_qa_output_name,
    resolve_qa_paths,
)


class QAPathsTest(unittest.TestCase):
    def test_named_output_separates_generations_from_benchmark_artifacts(self):
        paths = resolve_qa_paths(
            "/tmp/qa_benchmarks",
            "llava_1_5_7b",
            "VPEND-ablation",
            "pope",
        )

        self.assertEqual(
            paths.output_dir,
            Path("/tmp/qa_benchmarks/llava_1_5_7b/VPEND-ablation"),
        )
        self.assertEqual(paths.benchmark_dir, paths.output_dir / "pope")
        self.assertEqual(
            paths.generations_path,
            paths.output_dir / "pope_generations.jsonl",
        )
        self.assertEqual(
            paths.generation_failures_path,
            paths.output_dir / "pope_generation_failures.jsonl",
        )

    def test_each_benchmark_has_one_generation_file_in_shared_output_dir(self):
        root = "/tmp/qa_benchmarks"
        common = (root, "qwen3_vl_8b", "trial-01")
        datasets = ("pope", "clevr_exist_9k", "amber_discriminative")
        paths = [resolve_qa_paths(*common, dataset) for dataset in datasets]

        self.assertEqual(
            {item.output_dir for item in paths},
            {Path(root) / "qwen3_vl_8b" / "trial-01"},
        )
        self.assertEqual(
            {item.generations_path.name for item in paths},
            {
                "pope_generations.jsonl",
                "clevr_exist_9k_generations.jsonl",
                "amber_discriminative_generations.jsonl",
            },
        )

    def test_output_name_defaults_and_rejects_paths(self):
        self.assertEqual(
            resolve_qa_output_name(None, {}), DEFAULT_QA_OUTPUT
        )
        self.assertEqual(
            resolve_qa_output_name(None, {"output": "from-yaml"}),
            "from-yaml",
        )
        self.assertEqual(
            resolve_qa_output_name(
                "from-cli", {"output": "from-yaml"}
            ),
            "from-cli",
        )

        for invalid in ("", " ", ".", "..", "nested/name", r"nested\name"):
            with self.assertRaises(ValueError):
                resolve_qa_output_name(invalid)

    def test_generation_lookup_prefers_new_layout_and_reads_legacy(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            benchmark = Path(directory) / "trial" / "pope"
            benchmark.mkdir(parents=True)
            canonical = generations_path_for_benchmark_dir(benchmark)
            legacy = benchmark / "generations.jsonl"

            self.assertEqual(locate_qa_generations(benchmark), canonical)
            legacy.touch()
            self.assertEqual(locate_qa_generations(benchmark), legacy)
            canonical.touch()
            self.assertEqual(locate_qa_generations(benchmark), canonical)


if __name__ == "__main__":
    unittest.main()
