from __future__ import annotations

import math
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection.baselines import (
    DHCPMLP,
    SVARMLP,
    build_metatoken_classifier,
    evaluate_detection_scores,
    evaluate_hallucination_scores,
    raw_labels_to_hallucination_targets,
    select_detection_threshold,
    select_hallucination_threshold,
)
from features.baseline import (
    BaselineRuntime,
    DHCPShardReader,
    DHCPShardWriter,
    HalLocObjectDetector,
    attach_baseline,
    baseline_extraction_requirements,
    compute_metatoken_features,
    compute_metatoken_features_from_stats,
    compute_projectaway_internal_confidence,
    compute_svar_features,
    make_baseline_record,
    normalize_baseline_methods,
    resize_attention_preserve_mass,
    validate_baseline_record,
)
from models.base_wrapper import ModelOutput


class BaselineFeatureTests(unittest.TestCase):
    def test_baseline_requirements_do_not_retain_object_vocab_logits(self):
        meta = baseline_extraction_requirements(["metatoken"])
        self.assertFalse(meta.logits)
        self.assertTrue(meta.response_hidden_states)
        self.assertFalse(meta.patch_hidden_states)
        projectaway = baseline_extraction_requirements(["projectaway"])
        self.assertFalse(projectaway.logits)
        self.assertTrue(projectaway.patch_hidden_states)
        self.assertFalse(projectaway.response_hidden_states)

    def test_runtime_builds_all_baselines_from_shared_outputs(self):
        class _Wrapper:
            device = "cpu"
            model = nn.Identity()

        class _ClipCache:
            def encode(self, image):
                self.last_size = image.size
                return torch.arange(16, dtype=torch.float32).reshape(4, 4)

        self.assertEqual(
            set(normalize_baseline_methods("all")),
            {"metatoken", "svar", "dhcp", "projectaway", "halloc"},
        )
        response_ids = [1, 2, 3]
        compact = {
            "response_target_logprobs": torch.tensor([-1.0, -0.5, -0.25]),
            "response_target_probs": torch.exp(torch.tensor([-1.0, -0.5, -0.25])),
            "response_logprob_variances": torch.tensor([0.2, 0.3, 0.4]),
            "response_normalized_entropies": torch.tensor([0.8, 0.7, 0.6]),
            "response_top1_probs": torch.tensor([0.5, 0.6, 0.7]),
            "response_top2_probs": torch.tensor([0.2, 0.3, 0.4]),
        }
        attention = torch.full((2, 2, 4), 0.125)
        patch_hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) / 10
        # Qwen-family wrappers expose response hidden states as bfloat16.
        # The runtime must cast before NumPy serialization.
        response_hidden = torch.arange(15, dtype=torch.float32).reshape(3, 5).to(
            torch.bfloat16
        )
        outputs = [
            ModelOutput(
                token_id=response_ids[index],
                token_str=f"token-{index}",
                text_to_patch_attn=attention,
                text_to_text_attn=torch.empty(0),
                token_hidden_states=torch.empty(0),
                patch_hidden_states=patch_hidden,
                response_token_idx=index,
                visual_grid=(2, 2),
                response_hidden_states=response_hidden,
                baseline_capture=compact,
            )
            for index in (0, 2)
        ]
        spans = [
            {"word": "cat", "token_indices": [0], "label": 1},
            {"word": "cat", "token_indices": [2], "label": 0},
        ]
        output_layer = nn.Linear(3, 6, bias=True)
        with tempfile.TemporaryDirectory() as directory:
            with BaselineRuntime(
                wrapper=_Wrapper(),
                methods="all",
                baseline_dir=directory,
                config={"dhcp": {"shard_size": 8}},
                device="cpu",
                resume=False,
                clip_extractor=_ClipCache(),
                output_layer=output_layer,
                final_norm=nn.Identity(),
            ) as runtime:
                records = runtime.build_image_records(
                    image=Image.new("RGB", (8, 8)),
                    image_id=17,
                    response_token_ids=response_ids,
                    spans=spans,
                    model_outputs=outputs,
                )
                self.assertTrue(runtime.requirements.response_hidden_states)
                # The image transaction flushes DHCP before callers append
                # records, so a crash before runtime.close cannot leave a
                # published reference pointing at a missing shard.
                reader = DHCPShardReader(os.path.join(directory, "dhcp", "shards"))
                live_item = reader.load(
                    records[0]["baselines"]["dhcp"]["shard_reference"]
                )
                self.assertEqual(live_item.shape[-2:], (12, 12))
            self.assertEqual(len(records), 2)
            for record in records:
                validate_baseline_record(
                    record, required=normalize_baseline_methods("all")
                )
                self.assertEqual(
                    float(record["baselines"]["metatoken"]["vector"][1]), 2.0
                )
                self.assertTrue(
                    os.path.exists(
                        os.path.join(
                            directory, record["baselines"]["halloc"]["cache_file"]
                        )
                    )
                )
                with np.load(
                    os.path.join(
                        directory, record["baselines"]["halloc"]["cache_file"]
                    )
                ) as cache:
                    self.assertEqual(cache["lvlm_embeddings"].dtype, np.float16)
                    np.testing.assert_allclose(
                        cache["lvlm_embeddings"],
                        response_hidden.float().numpy(),
                    )
            reader = DHCPShardReader(os.path.join(directory, "dhcp", "shards"))
            item = reader.load(
                records[0]["baselines"]["dhcp"]["shard_reference"]
            )
            self.assertEqual(item.shape[-2:], (12, 12))

    def test_metatoken_matches_hand_computed_terms_and_compact_stats(self):
        ids = [0, 2, 1]
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0],
                [0.5, -0.5, 1.5, 0.0],
                [-1.0, 2.0, 0.0, 1.0],
            ]
        )
        attention = torch.tensor(
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.2, 0.4], [0.1, 0.5]],
            ]
        )
        result = compute_metatoken_features(
            response_token_ids=ids,
            token_logits=logits,
            visual_attention=attention,
            span_start=1,
            span_end=2,
            occurrence_count=2,
        )
        self.assertEqual(result.vector.shape, (12,))
        self.assertAlmostEqual(float(result.vector[0]), 1 / 3)
        self.assertAlmostEqual(float(result.vector[1]), 2.0)
        np.testing.assert_allclose(result.vector[2:4], [0.3, 0.3], rtol=1e-6)

        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        rows = torch.arange(3)
        target_log_probs = log_probs[rows, torch.tensor(ids)]
        self.assertAlmostEqual(
            float(result.vector[4]), float(target_log_probs[1:].sum()), places=6
        )
        self.assertAlmostEqual(
            float(result.vector[5]), float(target_log_probs.sum()), places=6
        )
        self.assertAlmostEqual(
            float(result.vector[-1]), float(probs[1, ids[1]]), places=6
        )

        top2 = torch.topk(probs, 2, dim=-1).values
        compact = compute_metatoken_features_from_stats(
            response_token_ids=ids,
            visual_attention=attention,
            span_start=1,
            span_end=2,
            occurrence_count=2,
            response_target_logprobs=target_log_probs,
            response_target_probs=probs[rows, torch.tensor(ids)],
            response_logprob_variances=log_probs.var(dim=-1, unbiased=False),
            response_normalized_entropies=-(probs * log_probs).sum(-1)
            / math.log(logits.shape[-1]),
            response_top1_probs=top2[:, 0],
            response_top2_probs=top2[:, 1],
        )
        np.testing.assert_allclose(compact.vector, result.vector, rtol=1e-6, atol=1e-7)

    def test_svar_keeps_layer_head_visual_attention_ratio(self):
        attention = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
        result = compute_svar_features(attention, layer_start=1, layer_end=3)
        expected = attention.sum(-1)[1:3]
        np.testing.assert_allclose(result.visual_attention_ratio, expected.numpy())
        np.testing.assert_allclose(result.vector, expected.reshape(-1).numpy())
        self.assertAlmostEqual(result.score, float(expected.mean(-1).sum()))

    def test_dhcp_resize_preserves_each_head_mass(self):
        attention = torch.rand(3, 2, 6)
        resized = resize_attention_preserve_mass(
            attention, source_grid=(2, 3), target_grid=(12, 12)
        )
        self.assertEqual(tuple(resized.shape), (3, 2, 12, 12))
        torch.testing.assert_close(
            resized.sum(dim=(-2, -1)), attention.sum(dim=-1), rtol=1e-5, atol=1e-6
        )
        zeros = resize_attention_preserve_mass(
            torch.zeros(1, 1, 4), source_grid=(2, 2)
        )
        self.assertEqual(float(zeros.sum()), 0.0)

    def test_dhcp_float16_shards_round_trip_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            first = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            with DHCPShardWriter(directory, shard_size=2) as writer:
                ref0 = writer.add(first)
                ref1 = writer.add(first + 1)
            reader = DHCPShardReader(directory)
            self.assertEqual(reader.load(ref0).dtype, np.float16)
            np.testing.assert_array_equal(reader.load(ref0), first.astype(np.float16))
            np.testing.assert_array_equal(reader.load(ref1), (first + 1).astype(np.float16))
            with DHCPShardWriter(directory, shard_size=1, resume=True) as writer:
                ref2 = writer.add(first + 2)
            self.assertNotEqual(ref0.shard, ref2.shard)
            np.testing.assert_array_equal(reader.load(ref2), (first + 2).astype(np.float16))

    def test_projectaway_chunked_softmax_is_exact(self):
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 1.0], [-1.0, 0.5]],
            ]
        )
        weight = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.5, -0.5]]
        )
        result = compute_projectaway_internal_confidence(
            hidden, weight, [1, 2], vocab_chunk_size=2, row_chunk_size=1
        )
        full = torch.softmax(hidden @ weight.T, dim=-1)[..., [1, 2]]
        expected_layers = full.amax(dim=(1, 2))
        np.testing.assert_allclose(
            result.per_layer_internal_confidence,
            expected_layers.numpy(),
            rtol=1e-6,
        )
        self.assertAlmostEqual(result.internal_confidence, float(full.max()), places=6)
        self.assertAlmostEqual(result.hallucination_score, 1.0 - float(full.max()), places=6)

    def test_schema_preserves_raw_labels(self):
        record = make_baseline_record(
            image_id=1,
            token_str="chair",
            response_token_idx=2,
            target_token_id=3,
            label=0,
        )
        attach_baseline(record, "svar", {"vector": np.array([1.0], np.float32)})
        validate_baseline_record(record, required=("svar",))
        self.assertEqual(record["label"], 0)
        self.assertEqual(record["label_semantics"]["0"], "hallucination")


class _FakeClip(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, *, pixel_values):
        batch = pixel_values.shape[0]
        return SimpleNamespace(
            last_hidden_state=torch.ones(batch, 3, 4, device=pixel_values.device)
            * self.scale
        )


class _FakeVisualBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6, visual_embedding_dim=5)

    def forward(self, **kwargs):
        text = kwargs["inputs_embeds"]
        # Retain text positions; the real model appends visual positions.
        return SimpleNamespace(last_hidden_state=text + 0.25)


class BaselineDetectorTests(unittest.TestCase):
    def test_halloc_injected_backbones_and_freeze_clip(self):
        clip = _FakeClip()
        model = HalLocObjectDetector(
            lvlm_hidden_size=8,
            clip_encoder=clip,
            visualbert=_FakeVisualBert(),
            freeze_clip=True,
        )
        model.train()
        self.assertFalse(clip.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in clip.parameters()))
        logits = model(
            lvlm_embeddings=torch.randn(2, 4, 8),
            pixel_values=torch.randn(2, 3, 16, 16),
            object_indices=torch.tensor([1, 3]),
        )
        self.assertEqual(tuple(logits.shape), (2, 2))
        self.assertEqual(model.paper_metadata()["heads"], ["object"])

    def test_detector_shapes_and_explicit_label_conversion(self):
        self.assertEqual(tuple(SVARMLP(10)(torch.randn(3, 10)).shape), (3, 2))
        self.assertEqual(tuple(DHCPMLP(12)(torch.randn(3, 2, 2, 3)).shape), (3, 2))
        raw = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(
            raw_labels_to_hallucination_targets(raw), [1, 0, 1, 0]
        )
        scores = np.array([0.9, 0.1, 0.8, 0.2])
        threshold = select_hallucination_threshold(raw, scores)
        metrics = evaluate_hallucination_scores(raw, scores, threshold)
        self.assertEqual(metrics["headline_positive_class"], "hallucination")
        self.assertEqual(metrics["hallucination_positive"]["f1"], 1.0)
        self.assertEqual(metrics["real_positive"]["f1"], 1.0)

        real_threshold = select_detection_threshold(
            raw,
            scores,
            positive_class="real",
        )
        real_metrics = evaluate_detection_scores(
            raw,
            scores,
            real_threshold,
            positive_class="real",
        )
        self.assertEqual(real_metrics["headline_positive_class"], "real")
        self.assertEqual(real_metrics["threshold_score_class"], "real")
        self.assertEqual(real_metrics["real_positive"]["f1"], 1.0)
        self.assertEqual(real_metrics["hallucination_positive"]["f1"], 1.0)

    def test_metatoken_classifier_paper_hyperparameters(self):
        lr = build_metatoken_classifier("lr").named_steps["classifier"]
        gb = build_metatoken_classifier("gb").named_steps["classifier"]
        self.assertEqual(lr.solver, "lbfgs")
        self.assertEqual(gb.n_estimators, 100)


if __name__ == "__main__":
    unittest.main()
