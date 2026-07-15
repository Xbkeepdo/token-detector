from detection.pope_paper_protocol import feature_vector, select_paper_samples


def test_paper_protocol_keeps_yes_and_flips_positive_class_to_hallucination():
    rows = [
        {"prediction": "yes", "label": 1, "ads_per_layer": [1], "object_cgc_per_layer": [2]},
        {"prediction": "yes", "label": 0, "ads_per_layer": [3], "object_cgc_per_layer": [4]},
        {"prediction": "no", "label": 0, "ads_per_layer": [5], "object_cgc_per_layer": [6]},
    ]
    selected = select_paper_samples(rows)
    assert [row["paper_label"] for row in selected] == [0, 1]
    assert feature_vector(selected[0], "ads+object_cgc").tolist() == [1.0, 2.0]
