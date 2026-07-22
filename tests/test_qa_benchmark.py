import json

from data.qa_benchmark import (
    _load_clevr_exist,
    _amber_dimension,
    assert_no_image_leakage,
    infer_clevr_query_object_span,
    label_answer,
    normalize_yes_no,
    question_key,
)


def test_answer_normalization_and_labels():
    assert normalize_yes_no(" Yes. Explanation") == "yes"
    assert normalize_yes_no("NO!") == "no"
    assert normalize_yes_no("nobody is present") is None
    assert label_answer("yes", "yes") == (1, "correct_yes")
    assert label_answer("no", "no") == (1, "correct_no")
    assert label_answer("yes", "no") == (0, "false_positive")
    assert label_answer("no", "yes") == (0, "false_negative")
    assert label_answer(None, "yes") == (0, "invalid")


def test_question_key_uses_dataset_source_and_id():
    row = {"dataset": "pope", "source_split": "random", "question_id": 7}
    assert question_key(row) == "pope::random::7"


def test_amber_dimensions_and_shared_image_namespace():
    assert _amber_dimension("discriminative-hallucination") == "existence"
    assert _amber_dimension("discriminative-attribute-state") == "attribute"
    assert _amber_dimension("discriminative-relation") == "relation"
    rows = [
        {
            "dataset": "amber_discriminative",
            "source_split": "existence",
            "image_id": 1,
            "probe_split": "train",
        },
        {
            "dataset": "amber_discriminative",
            "source_split": "relation",
            "image_id": 1,
            "probe_split": "test",
        },
    ]
    try:
        assert_no_image_leakage(rows)
    except ValueError:
        pass
    else:
        raise AssertionError("AMBER image leakage across dimensions was accepted")


def test_clevr_loader_records_the_configured_dataset_name(tmp_path):
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    payload = {
        "questions": [
            {
                "question_index": 7,
                "image_index": 3,
                "image_filename": "CLEVR_train_000003.png",
                "question": "Are there any spheres?",
                "answer": "yes",
                "question_family_index": 1,
                "program": [
                    {"function": "scene", "inputs": []},
                    {
                        "function": "filter_shape",
                        "inputs": [0],
                        "value_inputs": ["sphere"],
                    },
                    {"function": "exist", "inputs": [1]},
                ],
            }
        ]
    }
    (questions_dir / "CLEVR_train_questions.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    rows = _load_clevr_exist(tmp_path, "train", "clevr_exist_9k")
    assert rows[0]["dataset"] == "clevr_exist_9k"
    assert rows[0]["key"] == "clevr_exist_9k::train::7"


def test_clevr_image_identity_includes_official_source_split():
    rows = [
        {"source_split": "train", "image_id": 1, "probe_split": "train"},
        {"source_split": "val", "image_id": 1, "probe_split": "val"},
        {"source_split": "val", "image_id": 2, "probe_split": "test"},
    ]
    assert_no_image_leakage(rows)


def test_clevr_query_object_span_uses_entity_after_final_existential_trigger():
    program = [
        {"function": "scene", "inputs": []},
        {"function": "filter_shape", "inputs": [0], "value_inputs": ["sphere"]},
        {"function": "exist", "inputs": [1]},
    ]
    question = "There is a gray block; are there any spheres to the left of it?"
    result = infer_clevr_query_object_span(question, program)
    assert result["status"] == "found"
    assert result["surface"] == "spheres"
    assert question[result["char_start"]:result["char_end"]] == "spheres"


def test_clevr_generic_query_head_does_not_choose_later_reference_object():
    program = [
        {"function": "scene", "inputs": []},
        {"function": "exist", "inputs": [0]},
    ]
    question = (
        "Are there any other things that are the same shape as the big "
        "metallic object?"
    )
    result = infer_clevr_query_object_span(question, program)
    assert result["status"] == "found"
    assert result["surface"] == "things"


def test_clevr_object_span_is_explicitly_ambiguous_without_existential_trigger():
    result = infer_clevr_query_object_span(
        "Count the cubes.",
        [{"function": "exist", "inputs": []}],
    )
    assert result["status"] == "ambiguous"
    assert result["reason"] == "no_existential_trigger"
