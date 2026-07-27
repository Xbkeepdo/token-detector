from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.token_alignment import (  # noqa: E402
    build_response_token_offsets,
    locate_first_token_id,
    token_indices_for_char_span,
    validate_token_surface,
)


def _load_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Pluralizer:
    def plural(self, value: str) -> str:
        return {
            "person": "people",
            "officer": "officers",
            "sofa": "sofas",
            "couch": "couches",
            "dog": "dogs",
        }.get(value, f"{value}s")


class _FastTokenizer:
    is_fast = True
    all_special_ids = [99]

    def __init__(self) -> None:
        self.caption = "An officer sees another officer and a sofa."
        self.caption_ids = [10, 11, 12, 13, 11, 14, 15, 16, 17]
        self.caption_offsets = [
            (0, 2),
            (2, 10),
            (10, 15),
            (15, 23),
            (23, 31),
            (31, 35),
            (35, 37),
            (37, 42),
            (42, 43),
        ]
        self.pieces = {
            10: "An",
            11: " officer",
            12: " sees",
            13: " another",
            14: " and",
            15: " a",
            16: " sofa",
            17: ".",
            20: "officer",
            21: "officers",
            22: "person",
            23: "people",
            24: "sofa",
            25: "sofas",
            26: "couch",
            27: "couches",
            30: "dog",
            31: "dogs",
            99: "",
        }
        self.query_ids = {
            "officer": [20],
            "officers": [21],
            "person": [22],
            "people": [23],
            "sofa": [24],
            "sofas": [25],
            "couch": [26],
            "couches": [27],
            "dog": [30],
            "dogs": [31],
        }

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens
        if text == self.caption:
            result = {"input_ids": list(self.caption_ids)}
            if return_offsets_mapping:
                result["offset_mapping"] = list(self.caption_offsets)
            return result
        return {"input_ids": list(self.query_ids.get(text, [88]))}

    def encode(self, text: str, add_special_tokens: bool = False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        values = []
        for token_id in token_ids:
            if skip_special_tokens and int(token_id) in self.all_special_ids:
                continue
            values.append(self.pieces.get(int(token_id), "<?>"))
        return "".join(values)

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


class _SentencePieceTokenizer:
    is_fast = False
    all_special_ids = [99]

    class _Processor:
        @staticmethod
        def decode_ids_as_immutable_proto(token_ids):
            assert token_ids == [1, 2, 3]
            return SimpleNamespace(
                text="give me phone",
                pieces=[
                    SimpleNamespace(begin=0, end=4),
                    SimpleNamespace(begin=4, end=7),
                    SimpleNamespace(begin=7, end=13),
                ],
            )

    sp_model = _Processor()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        pieces = {1: "give", 2: " me", 3: " phone", 99: ""}
        return "".join(
            pieces[int(value)]
            for value in token_ids
            if not (skip_special_tokens and int(value) == 99)
        )


class _ZeroWidthSentencePieceTokenizer:
    is_fast = False
    all_special_ids = []

    class _Processor:
        @staticmethod
        def decode_ids_as_immutable_proto(token_ids):
            assert token_ids == [1, 2, 3, 4, 5]
            return SimpleNamespace(
                text="🙂 phone",
                pieces=[
                    SimpleNamespace(begin=0, end=0),
                    SimpleNamespace(begin=0, end=0),
                    SimpleNamespace(begin=0, end=0),
                    SimpleNamespace(begin=0, end=1),
                    SimpleNamespace(begin=1, end=7),
                ],
            )

    sp_model = _Processor()

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        values = tuple(int(value) for value in token_ids)
        decoded = {
            (): "",
            (1,): "�",
            (1, 2): "�",
            (1, 2, 3): "�",
            (1, 2, 3, 4): "🙂",
            (1, 2, 3, 4, 5): "🙂 phone",
            (5,): " phone",
        }
        return decoded.get(values, "�")


class _IncompleteSpecialIdTokenizer:
    """Mimic Llama-3 eot metadata missing from all_special_ids."""

    is_fast = False
    all_special_ids = [99]
    added_tokens_decoder = {
        99: SimpleNamespace(special=True),
        100: SimpleNamespace(special=True),
    }

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        pieces = {1: "A", 2: " dog", 3: ".", 99: "<eos>", 100: "<|eot_id|>"}
        return "".join(
            pieces[int(value)]
            for value in token_ids
            if not (
                skip_special_tokens and int(value) in {99, 100}
            )
        )

    def batch_decode(self, rows, **kwargs):
        return [self.decode(row, **kwargs) for row in rows]


class CocoTokenAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.label_coco = _load_module(
            "label_coco_alignment_tests",
            "coco-labeling/label_coco.py",
        )
        cls.coco_chair = _load_module(
            "coco_chair_alignment_tests",
            "coco-labeling/coco_chair.py",
        )

    def setUp(self) -> None:
        self.tokenizer = _FastTokenizer()
        self.label_coco._INFLECT_ENGINE = _Pluralizer()

    def test_fast_offsets_keep_actual_response_indices_and_ignore_eos(self) -> None:
        ids = self.tokenizer.caption_ids + [99]
        offsets = build_response_token_offsets(
            self.tokenizer,
            ids,
            self.tokenizer.caption,
        )
        indices = token_indices_for_char_span(offsets, 23, 31)
        self.assertEqual(indices, [4])
        self.assertIsNone(offsets[-1])
        validate_token_surface(
            tokenizer=self.tokenizer,
            response_token_ids=ids,
            token_indices=indices,
            offsets=offsets,
            caption=self.tokenizer.caption,
            char_start=23,
            char_end=31,
        )

    def test_sentencepiece_proto_retains_predictor_target_index(self) -> None:
        tokenizer = _SentencePieceTokenizer()
        ids = [1, 2, 3, 99]
        caption = "give me phone"
        offsets = build_response_token_offsets(tokenizer, ids, caption)
        phone_indices = token_indices_for_char_span(offsets, 8, 13)
        self.assertEqual(phone_indices, [2])
        # response index 2 is the target phone token; causal forward therefore
        # consumes ids[:2] and uses the final "me" state to predict it.
        self.assertEqual(ids[: phone_indices[0]], [1, 2])

    def test_sentencepiece_zero_width_unicode_pieces_use_decoder_fallback(self) -> None:
        tokenizer = _ZeroWidthSentencePieceTokenizer()
        ids = [1, 2, 3, 4, 5]
        caption = "🙂 phone"
        offsets = build_response_token_offsets(tokenizer, ids, caption)
        emoji_indices = token_indices_for_char_span(offsets, 0, 1)
        phone_indices = token_indices_for_char_span(offsets, 2, 7)
        self.assertEqual(emoji_indices, [0, 1, 2, 3])
        self.assertEqual(phone_indices, [4])
        validate_token_surface(
            tokenizer=tokenizer,
            response_token_ids=ids,
            token_indices=emoji_indices,
            offsets=offsets,
            caption=caption,
            char_start=0,
            char_end=1,
        )
        validate_token_surface(
            tokenizer=tokenizer,
            response_token_ids=ids,
            token_indices=phone_indices,
            offsets=offsets,
            caption=caption,
            char_start=2,
            char_end=7,
        )

    def test_added_special_eot_missing_from_all_special_ids_is_ignored(self) -> None:
        tokenizer = _IncompleteSpecialIdTokenizer()
        offsets = build_response_token_offsets(
            tokenizer,
            [1, 2, 3, 100],
            "A dog.",
        )
        self.assertEqual(offsets[:3], [(0, 1), (1, 5), (5, 6)])
        self.assertIsNone(offsets[3])

    def test_multitoken_unicode_span_preserves_bos_and_eos_indices(self) -> None:
        tokenizer = _FastTokenizer()
        tokenizer.all_special_ids = [98, 99]
        tokenizer.caption = "  A café has a cell phone, 中文."
        tokenizer.caption_ids = [40, 41, 42, 43, 44, 45, 46, 47, 48]
        tokenizer.caption_offsets = [
            (0, 3),
            (3, 8),
            (8, 12),
            (12, 14),
            (14, 19),
            (19, 25),
            (25, 26),
            (26, 29),
            (29, 30),
        ]
        tokenizer.pieces.update(
            {
                40: "  A",
                41: " café",
                42: " has",
                43: " a",
                44: " cell",
                45: " phone",
                46: ",",
                47: " 中文",
                48: ".",
                98: "",
            }
        )
        response_ids = [98] + tokenizer.caption_ids + [99]
        offsets = build_response_token_offsets(
            tokenizer,
            response_ids,
            tokenizer.caption,
        )
        indices = token_indices_for_char_span(offsets, 15, 25)
        self.assertEqual(indices, [5, 6])
        validate_token_surface(
            tokenizer=tokenizer,
            response_token_ids=response_ids,
            token_indices=indices,
            offsets=offsets,
            caption=tokenizer.caption,
            char_start=15,
            char_end=25,
        )

    def test_official_plural_fallback_uses_first_matching_token_id(self) -> None:
        location = locate_first_token_id(
            tokenizer=self.tokenizer,
            response_token_ids=[7, 31, 31],
            query="dog",
            pluralize=_Pluralizer().plural,
        )
        self.assertEqual(location["status"], "found")
        self.assertEqual(location["token_indices"], [1])
        self.assertTrue(location["used_plural_fallback"])
        self.assertEqual(location["matched_query"], "dogs")

    def test_v2_keeps_full_mentions_and_dedupes_training_samples(self) -> None:
        caption = self.tokenizer.caption
        chair_info = {
            "object_mentions": [
                {
                    "surface": "officer",
                    "normalized_word": "officer",
                    "canonical_object": "person",
                    "word_idx": 1,
                    "char_start": 3,
                    "char_end": 10,
                    "label": 1,
                },
                {
                    "surface": "officer",
                    "normalized_word": "officer",
                    "canonical_object": "person",
                    "word_idx": 4,
                    "char_start": 24,
                    "char_end": 31,
                    "label": 1,
                },
                {
                    "surface": "sofa",
                    "normalized_word": "sofa",
                    "canonical_object": "couch",
                    "word_idx": 7,
                    "char_start": 38,
                    "char_end": 42,
                    "label": 0,
                },
            ],
            "mscoco_generated_words": ["person", "person", "couch"],
            "mscoco_gt_words": ["person"],
            "mscoco_hallucinated_words": [("sofa", "couch")],
            "metrics": {"CHAIRs": 1, "CHAIRi": 1 / 3},
        }
        token_ids = self.tokenizer.caption_ids + [99]
        spans = self.label_coco._chair_token_spans(
            evaluator=None,
            tokenizer=self.tokenizer,
            image_id=7,
            caption=caption,
            token_ids=token_ids,
            chair_info=chair_info,
        )
        self.assertEqual([span["token_indices"] for span in spans], [[1], [4], [7]])
        self.assertEqual(spans[0]["occurrence_count"], 2)
        self.assertEqual(spans[1]["occurrence_index"], 2)
        self.assertEqual(
            set(spans[0]["token_locations"]),
            {
                "exact_response_offsets",
                "svar_surface_first_token_id",
                "svar_canonical_first_token_id",
            },
        )

        official = self.label_coco._build_official_svar_samples(
            tokenizer=self.tokenizer,
            token_ids=token_ids,
            chair_info=chair_info,
        )
        entry = self.label_coco._compact_label_entry(
            image_id=7,
            caption=caption,
            spans=spans,
            chair_info=chair_info,
            official_svar_samples=official,
        )
        self.assertEqual(entry["schema_version"], 2)
        self.assertEqual(len(entry["all_object_token_spans"]), 3)
        self.assertEqual(len(entry["object_token_spans"]), 2)
        self.assertEqual(
            [span["canonical_object"] for span in entry["object_token_spans"]],
            ["person", "couch"],
        )
        self.assertEqual(
            {item["search_term"] for item in entry["official_svar_samples"]},
            {"person", "sofa", "couch"},
        )
        person = next(
            item
            for item in entry["official_svar_samples"]
            if item["search_term"] == "person"
        )
        self.assertEqual(person["status"], "not_found")

        summary = self.coco_chair.chair_summary({"7": entry})
        self.assertEqual(summary["object_mentions"], 3)
        self.assertEqual(summary["hallucinated_mentions"], 1)
        self.assertAlmostEqual(summary["chair_i"], 1 / 3)

        all_mentions_entry = self.label_coco._compact_label_entry(
            image_id=7,
            caption=caption,
            spans=spans,
            chair_info=chair_info,
            official_svar_samples=official,
            sample_unit="all_mentions",
        )
        self.assertEqual(
            all_mentions_entry["labeling_protocol"]["sample_unit"],
            "all_mentions",
        )
        self.assertEqual(len(all_mentions_entry["object_token_spans"]), 3)

    def test_disabled_generation_manifest_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = self.label_coco._validate_or_initialize_generation_run_manifest(
                output_dir=output,
                model_key="test_model",
                model_cfg={"hf_name": "test/model", "max_new_tokens": 8},
                prompt="Describe this image.",
                generations={
                    "1": {
                        "generated_text": "a dog",
                        "response_token_ids": [1, 2],
                    }
                },
                generation_shard_dir=output / "generation_shards",
                expected_image_ids={1},
                fresh_start=False,
                adopt_legacy=False,
                validate_manifest=False,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertFalse((output / "generation_manifest.json").exists())

    def test_skip_keeps_full_mentions_but_never_substitutes_second_mention(self) -> None:
        spans = [
            {
                "word": "person",
                "canonical_object": "person",
                "word_idx": 0,
                "char_start": 0,
                "char_end": 7,
                "token_indices": [],
                "token_locations": {
                    "exact_response_offsets": {"status": "not_found"}
                },
            },
            {
                "word": "person",
                "canonical_object": "person",
                "word_idx": 2,
                "char_start": 12,
                "char_end": 19,
                "token_indices": [3],
                "token_locations": {
                    "exact_response_offsets": {"status": "found"}
                },
            },
            {
                "word": "couch",
                "canonical_object": "couch",
                "word_idx": 4,
                "char_start": 24,
                "char_end": 29,
                "token_indices": [5],
                "token_locations": {
                    "exact_response_offsets": {"status": "found"}
                },
            },
        ]
        selected = self.label_coco._first_canonical_mentions(spans)
        self.assertEqual(
            [span["canonical_object"] for span in selected],
            ["couch"],
        )

    def test_reuse_generation_artifacts_copies_without_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            target = root / "new"
            source.mkdir()
            (source / "generations.json").write_text(
                '{"1":{"generated_text":"a dog","response_token_ids":[1,2]}}',
                encoding="utf-8",
            )
            (source / "image_splits.json").write_text(
                '{"train":[1],"val":[],"test":[]}',
                encoding="utf-8",
            )
            model_cfg = {"hf_name": "test/model", "max_new_tokens": 8}
            generation_payload = self.label_coco.load_json(
                str(source / "generations.json")
            )
            self.label_coco.save_json(
                self.label_coco.build_generation_manifest(
                    model="test_model",
                    model_cfg=model_cfg,
                    prompt="Describe this image.",
                    generations=generation_payload,
                    expected_image_ids={1},
                ),
                str(source / "generation_manifest.json"),
            )
            self.label_coco._reuse_generation_artifacts(
                source_output=source,
                target_output=target,
                expected_image_ids={1},
                model_key="test_model",
                model_cfg=model_cfg,
                prompt="Describe this image.",
            )
            self.assertTrue((target / "generations.json").is_file())
            self.assertFalse((target / "generations.json").is_symlink())
            self.assertTrue((target / "image_splits.json").is_file())
            self.assertFalse((target / "image_splits.json").is_symlink())

    def test_reuse_refuses_downstream_overwrite_and_mismatched_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            target = root / "new"
            source.mkdir()
            target.mkdir()
            (source / "generations.json").write_text(
                "{\"1\":{\"generated_text\":\"a dog\",\"response_token_ids\":[1,2]}}",
                encoding="utf-8",
            )
            model_cfg = {"hf_name": "test/model", "max_new_tokens": 8}
            generation_payload = self.label_coco.load_json(
                str(source / "generations.json")
            )
            self.label_coco.save_json(
                self.label_coco.build_generation_manifest(
                    model="test_model",
                    model_cfg=model_cfg,
                    prompt="Describe this image.",
                    generations=generation_payload,
                    expected_image_ids={1},
                ),
                str(source / "generation_manifest.json"),
            )
            (target / "labeling.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "downstream artifacts"):
                self.label_coco._reuse_generation_artifacts(
                    source_output=source,
                    target_output=target,
                    expected_image_ids={1},
                    model_key="test_model",
                    model_cfg=model_cfg,
                    prompt="Describe this image.",
                    resume=False,
                )

            (target / "labeling.json").unlink()
            (target / "generations.json").write_text(
                "{\"1\":{\"generated_text\":\"a cat\",\"response_token_ids\":[1,3]}}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "differ from source"):
                self.label_coco._reuse_generation_artifacts(
                    source_output=source,
                    target_output=target,
                    expected_image_ids={1},
                    model_key="test_model",
                    model_cfg=model_cfg,
                    prompt="Describe this image.",
                    resume=True,
                )

    def test_unicode_case_expansion_does_not_shift_chair_offsets(self) -> None:
        evaluator = object.__new__(self.coco_chair.CocoChairEvaluator)
        evaluator._tokenizer = self.coco_chair.TreebankWordTokenizer()
        evaluator._lemmatizer = SimpleNamespace(
            lemmatize=lambda word, pos=None: word
        )
        evaluator.get_wordnet_pos = lambda tag: "n"
        evaluator.double_word_dict = {}
        with mock.patch.object(
            self.coco_chair.nltk,
            "pos_tag",
            side_effect=lambda words: [(word, "NN") for word in words],
        ):
            units = evaluator._caption_to_units("İ person")
        person = next(unit for unit in units if unit["word"] == "person")
        self.assertEqual((person["char_start"], person["char_end"]), (2, 8))
        self.assertEqual(person["surface"], "person")

    def test_partial_generation_manifest_rejects_prompt_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model_cfg = {"hf_name": "test/model", "max_new_tokens": 8}
            first = self.label_coco._validate_or_initialize_generation_run_manifest(
                output_dir=output,
                model_key="test_model",
                model_cfg=model_cfg,
                prompt="Describe this image.",
                generations={},
                generation_shard_dir=output / "generation_shards",
                expected_image_ids={1, 2},
                fresh_start=True,
                adopt_legacy=False,
            )
            self.assertEqual(first["status"], "in_progress")
            with self.assertRaisesRegex(ValueError, "model/prompt/cohort"):
                self.label_coco._validate_or_initialize_generation_run_manifest(
                    output_dir=output,
                    model_key="test_model",
                    model_cfg=model_cfg,
                    prompt="Use a different prompt.",
                    generations={},
                    generation_shard_dir=output / "generation_shards",
                    expected_image_ids={1, 2},
                    fresh_start=False,
                    adopt_legacy=False,
                )

    def test_complete_copied_generations_auto_write_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            model_cfg = {"hf_name": "test/model", "max_new_tokens": 8}
            generations = {
                "1": {
                    "generated_text": "a dog",
                    "response_token_ids": [1, 2],
                },
                "2": {
                    "generated_text": "a cat",
                    "response_token_ids": [3, 4],
                },
            }
            manifest = self.label_coco._validate_or_initialize_generation_run_manifest(
                output_dir=output,
                model_key="test_model",
                model_cfg=model_cfg,
                prompt="Describe this image.",
                generations=generations,
                generation_shard_dir=output / "generation_shards",
                expected_image_ids={1, 2},
                fresh_start=False,
                adopt_legacy=False,
            )
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(
                (output / "generation_manifest.json").is_file()
            )

    def test_schema_v2_never_reencodes_caption_as_generation_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "actual response_token_ids"):
            self.label_coco._generation_token_ids(
                {},
                9,
                "a dog",
                self.tokenizer,
            )

    def test_explicit_adoption_adds_only_generation_provenance(self) -> None:
        labeling = {
            "1": {
                "schema_version": 2,
                "image_id": 1,
                "generated_text": "a dog",
                "hallucinated_words": [],
                "real_words": [],
                "object_token_spans": [],
                "all_object_token_spans": [],
                "official_svar_samples": [],
                "chair_s": 0,
                "chair_i": 0.0,
            }
        }
        generations = {
            "1": {
                "generated_text": "a dog",
                "response_token_ids": [1, 2],
            }
        }
        expected = {
            "label_schema_version": 2,
            "sample_unit": "first_canonical_mention",
            "primary_locator": "exact_response_offsets",
            "generation_sha256": "generation-hash",
            "generation_provenance_sha256": "signed-generation-hash",
        }
        legacy = {
            key: value
            for key, value in expected.items()
            if key != "generation_provenance_sha256"
        }
        legacy["labeling_sha256"] = self.label_coco._json_sha256(labeling)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = Path(directory) / "labeling_manifest.json"
            self.label_coco.save_json(legacy, str(manifest_path))
            self.label_coco._validate_existing_labeling_for_resume(
                labeling=labeling,
                manifest_path=str(manifest_path),
                expected_manifest=expected,
                samples=[{"image_id": 1}],
                generations=generations,
                adopt_legacy=True,
            )
            migrated = self.label_coco.load_json(str(manifest_path))
            self.assertEqual(
                migrated["generation_provenance_sha256"],
                "signed-generation-hash",
            )

    def test_resume_rejects_legacy_labeling_without_v2_manifest(self) -> None:
        labeling = {
            "1": {
                "image_id": 1,
                "generated_text": "a dog",
                "object_token_spans": [],
            }
        }
        generations = {
            "1": {
                "generated_text": "a dog",
                "response_token_ids": [1, 2],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "legacy labeling.json"):
                self.label_coco._validate_existing_labeling_for_resume(
                    labeling=labeling,
                    manifest_path=str(Path(directory) / "labeling_manifest.json"),
                    expected_manifest={
                        "label_schema_version": 2,
                        "sample_unit": "first_canonical_mention",
                        "primary_locator": "exact_response_offsets",
                    },
                    samples=[{"image_id": 1}],
                    generations=generations,
                )


if __name__ == "__main__":
    unittest.main()
