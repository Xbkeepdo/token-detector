"""COCO/CHAIR-style object mention labeling for generated captions."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)

COCO_ALIASES: dict[str, list[str]] = {
    "person": ["person", "people", "man", "men", "woman", "women", "boy", "boys", "girl", "girls", "child", "children", "kid", "kids", "baby", "player", "players", "skier", "surfer"],
    "bicycle": ["bicycle", "bicycles", "bike", "bikes", "cycle", "cycles"],
    "car": ["car", "cars", "automobile", "automobiles", "taxi", "taxis", "sedan", "sedans", "suv", "suvs"],
    "motorcycle": ["motorcycle", "motorcycles", "motorbike", "motorbikes", "moped", "mopeds"],
    "airplane": ["airplane", "airplanes", "plane", "planes", "jet", "jets", "aircraft"],
    "bus": ["bus", "buses", "coach", "coaches"],
    "train": ["train", "trains", "locomotive", "locomotives"],
    "truck": ["truck", "trucks", "pickup", "pickups", "lorry", "lorries"],
    "boat": ["boat", "boats", "ship", "ships", "canoe", "canoes", "kayak", "kayaks"],
    "traffic light": ["traffic light", "traffic lights", "stoplight", "stoplights", "street light", "street lights"],
    "fire hydrant": ["fire hydrant", "fire hydrants", "hydrant", "hydrants"],
    "stop sign": ["stop sign", "stop signs"],
    "parking meter": ["parking meter", "parking meters"],
    "bench": ["bench", "benches"],
    "bird": ["bird", "birds", "duck", "ducks", "seagull", "seagulls"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "horse": ["horse", "horses", "pony", "ponies"],
    "sheep": ["sheep"],
    "cow": ["cow", "cows", "cattle"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "zebra": ["zebra", "zebras"],
    "giraffe": ["giraffe", "giraffes"],
    "backpack": ["backpack", "backpacks", "rucksack", "rucksacks"],
    "umbrella": ["umbrella", "umbrellas"],
    "handbag": ["handbag", "handbags", "purse", "purses"],
    "tie": ["tie", "ties", "necktie", "neckties", "bow tie", "bow ties"],
    "suitcase": ["suitcase", "suitcases", "luggage"],
    "frisbee": ["frisbee", "frisbees", "disc", "discs"],
    "skis": ["ski", "skis"],
    "snowboard": ["snowboard", "snowboards"],
    "sports ball": ["sports ball", "sports balls", "ball", "balls", "soccer ball", "football", "basketball", "tennis ball", "baseball"],
    "kite": ["kite", "kites"],
    "baseball bat": ["baseball bat", "baseball bats", "bat", "bats"],
    "baseball glove": ["baseball glove", "baseball gloves", "glove", "gloves", "mitt", "mitts"],
    "skateboard": ["skateboard", "skateboards"],
    "surfboard": ["surfboard", "surfboards"],
    "tennis racket": ["tennis racket", "tennis rackets", "racket", "rackets", "racquet", "racquets"],
    "bottle": ["bottle", "bottles"],
    "wine glass": ["wine glass", "wine glasses", "glass", "glasses"],
    "cup": ["cup", "cups", "mug", "mugs"],
    "fork": ["fork", "forks"],
    "knife": ["knife", "knives"],
    "spoon": ["spoon", "spoons"],
    "bowl": ["bowl", "bowls"],
    "banana": ["banana", "bananas"],
    "apple": ["apple", "apples"],
    "sandwich": ["sandwich", "sandwiches", "burger", "burgers", "hamburger", "hamburgers"],
    "orange": ["orange", "oranges"],
    "broccoli": ["broccoli"],
    "carrot": ["carrot", "carrots"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "pizza": ["pizza", "pizzas"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "cake": ["cake", "cakes"],
    "chair": ["chair", "chairs"],
    "couch": ["couch", "couches", "sofa", "sofas"],
    "potted plant": ["potted plant", "potted plants", "plant", "plants"],
    "bed": ["bed", "beds"],
    "dining table": ["dining table", "dining tables", "table", "tables"],
    "toilet": ["toilet", "toilets"],
    "tv": ["tv", "tvs", "television", "televisions", "monitor", "monitors"],
    "laptop": ["laptop", "laptops", "notebook", "notebooks"],
    "mouse": ["mouse", "mice"],
    "remote": ["remote", "remotes"],
    "keyboard": ["keyboard", "keyboards"],
    "cell phone": ["cell phone", "cell phones", "mobile phone", "mobile phones", "phone", "phones", "smartphone", "smartphones"],
    "microwave": ["microwave", "microwaves"],
    "oven": ["oven", "ovens", "stove", "stoves"],
    "toaster": ["toaster", "toasters"],
    "sink": ["sink", "sinks"],
    "refrigerator": ["refrigerator", "refrigerators", "fridge", "fridges"],
    "book": ["book", "books"],
    "clock": ["clock", "clocks"],
    "vase": ["vase", "vases"],
    "scissors": ["scissors"],
    "teddy bear": ["teddy bear", "teddy bears", "teddy", "teddies", "stuffed bear"],
    "hair drier": ["hair drier", "hair driers", "hair dryer", "hair dryers", "dryer", "dryers"],
    "toothbrush": ["toothbrush", "toothbrushes"],
}

IRREGULAR_SINGULARS = {
    "people": "person",
    "men": "man",
    "women": "woman",
    "children": "child",
    "mice": "mouse",
    "knives": "knife",
    "buses": "bus",
    "glasses": "glass",
    "teddies": "teddy",
}


class CocoChairEvaluator:
    """Small CHAIR-style evaluator backed by COCO instance annotations."""

    def __init__(self, instances_file: str | Path, captions_file: str | Path | None = None) -> None:
        self.instances_file = Path(instances_file)
        self.captions_file = Path(captions_file) if captions_file is not None else None
        self.alias_to_canonical = self._build_alias_map()
        self.double_aliases = {
            alias: canonical
            for alias, canonical in self.alias_to_canonical.items()
            if " " in alias
        }
        self.single_aliases = {
            alias: canonical
            for alias, canonical in self.alias_to_canonical.items()
            if " " not in alias
        }
        self.image_id_to_objects: dict[int, set[str]] = {}
        self.category_id_to_name: dict[int, str] = {}
        self.val_image_ids: list[int] = []
        self._load_annotations()

    @classmethod
    def from_cache(
        cls,
        instances_file: str | Path,
        captions_file: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> "CocoChairEvaluator":
        cache = Path(cache_path) if cache_path else None
        if cache is not None and cache.exists():
            with cache.open("rb") as handle:
                return pickle.load(handle)
        obj = cls(instances_file=instances_file, captions_file=captions_file)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as handle:
                pickle.dump(obj, handle)
        return obj

    def get_ground_truth_objects(self, image_id: int) -> set[str]:
        return set(self.image_id_to_objects.get(int(image_id), set()))

    def evaluate_caption(self, image_id: int, caption: str) -> dict[str, Any]:
        gt_objects = self.get_ground_truth_objects(image_id)
        mentions = self.caption_to_mentions(caption)
        for mention in mentions:
            mention["hallucinated"] = int(mention["canonical_name"] not in gt_objects)
        hallucinated = [m for m in mentions if m["hallucinated"]]
        recalled = [m for m in mentions if not m["hallucinated"]]
        return {
            "image_id": int(image_id),
            "caption": caption,
            "ground_truth_objects": sorted(gt_objects),
            "object_mentions": mentions,
            "hallucinated_mentions": hallucinated,
            "recall_mentions": recalled,
            "chair_s": 1.0 if hallucinated else 0.0,
            "chair_i": float(len(hallucinated) / len(mentions)) if mentions else 0.0,
        }

    def caption_to_mentions(self, caption: str) -> list[dict[str, Any]]:
        tokens = [
            {
                "surface": match.group(0),
                "norm": self._normalize_token(match.group(0).lower()),
                "char_start": match.start(),
                "char_end": match.end(),
            }
            for match in WORD_RE.finditer(caption)
        ]
        mentions = []
        i = 0
        while i < len(tokens):
            alias = None
            span = 1
            if i + 1 < len(tokens):
                two_word = f"{tokens[i]['norm']} {tokens[i + 1]['norm']}"
                if two_word in self.double_aliases:
                    alias = two_word
                    span = 2
            if alias is None and tokens[i]["norm"] in self.single_aliases:
                alias = tokens[i]["norm"]
            if alias is not None:
                canonical = self.alias_to_canonical[alias]
                char_start = int(tokens[i]["char_start"])
                char_end = int(tokens[i + span - 1]["char_end"])
                mentions.append(
                    {
                        "surface": caption[char_start:char_end],
                        "canonical_name": canonical,
                        "char_start": char_start,
                        "char_end": char_end,
                        "token_start": i,
                        "token_end": i + span,
                    }
                )
                i += span
            else:
                i += 1
        return mentions

    def _load_annotations(self) -> None:
        with self.instances_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.category_id_to_name = {
            int(category["id"]): str(category["name"])
            for category in payload.get("categories", [])
        }
        self.val_image_ids = sorted(int(item["id"]) for item in payload.get("images", []))
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            category_name = self.category_id_to_name.get(int(annotation["category_id"]), "")
            canonical = self.alias_to_canonical.get(category_name, category_name)
            if canonical:
                self.image_id_to_objects.setdefault(image_id, set()).add(canonical)

    def _build_alias_map(self) -> dict[str, str]:
        mapping = {}
        for canonical, aliases in COCO_ALIASES.items():
            mapping[canonical] = canonical
            for alias in aliases:
                mapping[alias] = canonical
        return mapping

    def _normalize_token(self, token: str) -> str:
        if token in IRREGULAR_SINGULARS:
            return IRREGULAR_SINGULARS[token]
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ves") and len(token) > 4:
            return token[:-3] + "f"
        if token.endswith("es") and len(token) > 3 and not token.endswith(("ses", "xes", "zes")):
            return token[:-2]
        if token.endswith("s") and len(token) > 3 and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        return token


def dedupe_mentions_by_object(mentions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first mention per canonical COCO object in caption order."""
    seen = set()
    result = []
    for mention in sorted(mentions, key=lambda item: int(item.get("char_start", 0))):
        key = str(mention.get("canonical_name", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(mention))
    return result


def chair_summary(labeling: dict[str, dict] | dict[int, dict]) -> dict[str, float | int]:
    total_images = len(labeling)
    total_mentions = 0
    hallucinated_mentions = 0
    hallucinated_images = 0
    for item in labeling.values():
        spans = item.get("object_token_spans", [])
        hall = sum(1 for span in spans if int(span.get("label", 0)) == 1)
        total_mentions += len(spans)
        hallucinated_mentions += hall
        hallucinated_images += int(hall > 0)
    return {
        "images": total_images,
        "object_mentions": total_mentions,
        "hallucinated_mentions": hallucinated_mentions,
        "images_with_hallucination": hallucinated_images,
        "chair_s": hallucinated_images / total_images if total_images else 0.0,
        "chair_i": hallucinated_mentions / total_mentions if total_mentions else 0.0,
    }


def iter_ground_truth_entries(
    evaluator: CocoChairEvaluator,
    image_ids: Iterable[int],
) -> Iterable[dict[str, Any]]:
    for image_id in image_ids:
        yield {
            "image_id": int(image_id),
            "image": f"COCO_val2014_{int(image_id):012d}.jpg",
            "objects": sorted(evaluator.get_ground_truth_objects(int(image_id))),
        }
