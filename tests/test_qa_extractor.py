from features.qa_extractor import JSONLCheckpointStore, find_answer_semantic_token, qa_prompt


class TinyTokenizer:
    table = {1: "\n", 2: " Yes", 3: ".", 4: "nobody"}

    def encode(self, text, add_special_tokens=False):
        mapping = {"yes": [2], " yes": [2], "Yes": [2], " Yes": [2]}
        return mapping.get(text, [])

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.table.get(item, "") for item in ids)


def test_find_first_answer_semantic_token():
    assert find_answer_semantic_token([1, 2, 3], TinyTokenizer(), "yes") == 1
    assert find_answer_semantic_token([4], TinyTokenizer(), None) == 0


def test_model_specific_prompt_format():
    assert qa_prompt("llava_1_5_7b", "Is there a cat?").startswith("USER: <image>")
    assert qa_prompt("qwen2_5_vl_7b", "Is there a cat?") == "Is there a cat?\nAnswer only yes or no."


def test_jsonl_resume_does_not_rewrite_identical_row(tmp_path):
    path = tmp_path / "rows.jsonl"
    store = JSONLCheckpointStore(str(path), checkpoint_every=1)
    store.add({"key": "x", "value": 1})
    first_mtime = path.stat().st_mtime_ns
    resumed = JSONLCheckpointStore(str(path), checkpoint_every=1)
    resumed.add({"key": "x", "value": 1})
    assert resumed.dirty == 0
    assert path.stat().st_mtime_ns == first_mtime
