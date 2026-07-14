from data.qa_benchmark import assert_no_image_leakage, label_answer, normalize_yes_no, question_key


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


def test_clevr_image_identity_includes_official_source_split():
    rows = [
        {"source_split": "train", "image_id": 1, "probe_split": "train"},
        {"source_split": "val", "image_id": 1, "probe_split": "val"},
        {"source_split": "val", "image_id": 2, "probe_split": "test"},
    ]
    assert_no_image_leakage(rows)
