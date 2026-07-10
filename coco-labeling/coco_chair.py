"""COCO/CHAIR-style object mention labeling for generated captions."""

from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

LABEL_HALLUCINATED = 0
LABEL_REAL = 1
LABEL_IGNORE = -100
CHAIR_CACHE_VERSION = 2

# Copied from the standalone CHAIR evaluator used by
# ZhangqiJiang07/middle_layers_indicating_hallucinations.
synonyms_txt = """
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
"""

COCO_DOUBLE_WORDS = [
    "motor bike",
    "motor cycle",
    "air plane",
    "traffic light",
    "street light",
    "traffic signal",
    "stop light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "suit case",
    "sports ball",
    "baseball bat",
    "baseball glove",
    "tennis racket",
    "wine glass",
    "hot dog",
    "cell phone",
    "mobile phone",
    "teddy bear",
    "hair drier",
    "potted plant",
    "bow tie",
    "laptop computer",
    "stove top oven",
    "home plate",
    "train track",
]

ANIMAL_WORDS = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "animal",
    "cub",
]
VEHICLE_WORDS = ["jet", "train"]

NAMES_TO_RESOURCES = {
    "punkt": ("tokenizers/punkt", "tokenizers/punkt.zip"),
    "averaged_perceptron_tagger": (
        "taggers/averaged_perceptron_tagger",
        "taggers/averaged_perceptron_tagger.zip",
    ),
    "wordnet": ("corpora/wordnet", "corpora/wordnet.zip"),
    "omw-1.4": ("corpora/omw-1.4", "corpora/omw-1.4.zip"),
}


def ensure_nltk_data(download: bool = False) -> None:
    """Validate NLTK data needed by CHAIR tokenization and lemmatization."""
    missing = []
    for package, resources in NAMES_TO_RESOURCES.items():
        if not _has_any_nltk_resource(resources):
            missing.append(package)

    if missing and download:
        for package in missing:
            nltk.download(package, quiet=True)
        missing = []
        for package, resources in NAMES_TO_RESOURCES.items():
            if not _has_any_nltk_resource(resources):
                missing.append(package)

    if missing:
        raise LookupError(
            "Missing NLTK resources for CHAIR labeling: "
            f"{', '.join(missing)}. Run: python -m nltk.downloader "
            + " ".join(missing)
        )


def _has_any_nltk_resource(resources: Sequence[str]) -> bool:
    for resource in resources:
        try:
            nltk.data.find(resource)
            return True
        except LookupError:
            continue
    return False


def combine_coco_captions(annotation_path: str | Path) -> dict[str, Any]:
    annotation_path = Path(annotation_path)
    val_path = annotation_path / "captions_val2014.json"
    train_path = annotation_path / "captions_train2014.json"
    if not val_path.exists() or not train_path.exists():
        raise FileNotFoundError(
            "Expected captions_val2014.json and captions_train2014.json under "
            f"{annotation_path}"
        )
    val_caps = _load_json(val_path)
    train_caps = _load_json(train_path)
    return {
        "info": train_caps.get("info"),
        "licenses": train_caps.get("licenses"),
        "images": val_caps.get("images", []) + train_caps.get("images", []),
        "annotations": val_caps.get("annotations", []) + train_caps.get("annotations", []),
    }


def combine_coco_instances(annotation_path: str | Path) -> dict[str, Any]:
    annotation_path = Path(annotation_path)
    val_path = annotation_path / "instances_val2014.json"
    train_path = annotation_path / "instances_train2014.json"
    if not val_path.exists() or not train_path.exists():
        raise FileNotFoundError(
            "Expected instances_val2014.json and instances_train2014.json under "
            f"{annotation_path}"
        )
    val_instances = _load_json(val_path)
    train_instances = _load_json(train_path)
    return {
        "info": train_instances.get("info"),
        "licenses": train_instances.get("licenses"),
        "type": train_instances.get("type"),
        "categories": train_instances.get("categories", []),
        "images": train_instances.get("images", []) + val_instances.get("images", []),
        "annotations": val_instances.get("annotations", [])
        + train_instances.get("annotations", []),
    }


class CHAIR:
    """CHAIR object mention evaluator backed by COCO instances and captions."""

    def __init__(
        self,
        coco_path: str | Path | None = None,
        *,
        instances_file: str | Path | None = None,
        captions_file: str | Path | None = None,
    ) -> None:
        ensure_nltk_data(download=False)
        self.cache_version = CHAIR_CACHE_VERSION
        self.coco_path = Path(coco_path) if coco_path is not None else None
        self.instances_file = Path(instances_file) if instances_file is not None else None
        self.captions_file = Path(captions_file) if captions_file is not None else None
        self.imid_to_objects: dict[int, set[str]] = defaultdict(set)
        self.val_image_ids: list[int] = []
        self._tokenizer = TreebankWordTokenizer()
        self._lemmatizer = WordNetLemmatizer()

        self.synonym_groups = _parse_synonyms(synonyms_txt)
        self.mscoco_objects: list[str] = []
        self.inverse_synonym_dict: dict[str, str] = {}
        for synonym_group in self.synonym_groups:
            if not synonym_group:
                continue
            canonical = synonym_group[0]
            for synonym in synonym_group:
                self.mscoco_objects.append(synonym)
                self.inverse_synonym_dict[synonym] = canonical
        self.mscoco_object_set = set(self.mscoco_objects)
        self.double_word_dict = self._build_double_word_dict()
        self.get_annotations()

    @classmethod
    def from_cache(
        cls,
        instances_file: str | Path,
        captions_file: str | Path | None = None,
        cache_path: str | Path | None = None,
    ) -> "CHAIR":
        cache = Path(cache_path) if cache_path else None
        if cache is not None and cache.exists():
            try:
                with cache.open("rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, cls) and _cache_is_compatible(cached):
                    return cached
            except Exception as exc:
                print(f"[COCO-CHAIR] Ignoring incompatible CHAIR cache {cache}: {exc}")
        obj = cls(instances_file=instances_file, captions_file=captions_file)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp_cache = cache.with_name(f"{cache.name}.tmp")
            try:
                with tmp_cache.open("wb") as handle:
                    pickle.dump(obj, handle)
                tmp_cache.replace(cache)
            except Exception:
                tmp_cache.unlink(missing_ok=True)
                raise
        return obj

    def get_wordnet_pos(self, tag: str) -> str | None:
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    def caption_to_words(
        self,
        caption: str,
    ) -> tuple[list[str], list[str], list[int], list[str]]:
        """Return CHAIR-style COCO mention words, canonical objects, indices, and words."""
        units = self._caption_to_units(caption)
        object_units = [unit for unit in units if unit["word"] in self.mscoco_object_set]
        words = [str(unit["word"]) for unit in object_units]
        node_words = [self.inverse_synonym_dict[word] for word in words]
        idxs = [int(unit["word_idx"]) for unit in object_units]
        raw_words = [str(unit["word"]) for unit in units]
        return words, node_words, idxs, raw_words

    def caption_to_mentions(self, caption: str) -> list[dict[str, Any]]:
        units = self._caption_to_units(caption)
        mentions = []
        for unit in units:
            word = str(unit["word"])
            if word not in self.mscoco_object_set:
                continue
            canonical = self.inverse_synonym_dict[word]
            mentions.append(
                {
                    "surface": unit["surface"],
                    "surface_word": unit["surface"],
                    "normalized_word": word,
                    "canonical_name": canonical,
                    "canonical_object": canonical,
                    "word_idx": int(unit["word_idx"]),
                    "char_start": int(unit["char_start"]),
                    "char_end": int(unit["char_end"]),
                    "token_start": int(unit["word_idx"]),
                    "token_end": int(unit["word_idx"]) + int(unit["word_span"]),
                }
            )
        return mentions

    def get_annotations_from_segments(self) -> None:
        coco_segments = self._load_instances_payload()
        id_to_name = {
            int(category["id"]): _normalise_phrase(str(category["name"]))
            for category in coco_segments.get("categories", [])
        }
        self.val_image_ids = sorted(
            int(image["id"]) for image in coco_segments.get("images", []) if "id" in image
        )
        for annotation in coco_segments.get("annotations", []):
            image_id = int(annotation["image_id"])
            category_name = id_to_name.get(int(annotation.get("category_id", -1)), "")
            node_word = self.inverse_synonym_dict.get(category_name, category_name)
            if node_word:
                self.imid_to_objects[image_id].add(node_word)

    def get_annotations_from_captions(self) -> None:
        coco_caps = self._load_captions_payload()
        if not coco_caps:
            return
        for annotation in coco_caps.get("annotations", []):
            image_id = int(annotation["image_id"])
            _words, node_words, _idxs, _raw_words = self.caption_to_words(
                str(annotation.get("caption", ""))
            )
            self.imid_to_objects[image_id].update(node_words)

    def get_annotations(self) -> None:
        self.get_annotations_from_segments()
        self.get_annotations_from_captions()
        self.imid_to_objects = {
            int(image_id): set(objects)
            for image_id, objects in self.imid_to_objects.items()
        }

    def get_ground_truth_objects(self, image_id: int) -> set[str]:
        return set(self.imid_to_objects.get(int(image_id), set()))

    def evaluate_caption(self, image_id: int | str, caption: str) -> dict[str, Any]:
        cap_dict = self.compute_chair_token(image_id, caption)
        hallucinated = [
            mention for mention in cap_dict["object_mentions"]
            if int(mention["label"]) == LABEL_HALLUCINATED
        ]
        real = [
            mention for mention in cap_dict["object_mentions"]
            if int(mention["label"]) == LABEL_REAL
        ]
        return {
            "image_id": cap_dict["image_id"],
            "caption": caption,
            "ground_truth_objects": cap_dict["mscoco_gt_words"],
            "object_mentions": cap_dict["object_mentions"],
            "hallucinated_mentions": hallucinated,
            "recall_mentions": real,
            "chair_s": cap_dict["metrics"]["CHAIRs"],
            "chair_i": cap_dict["metrics"]["CHAIRi"],
            "metrics": cap_dict["metrics"],
            "word_labels": cap_dict["word_labels"],
        }

    def compute_chair_token(self, image_id: int | str, caption: str) -> dict[str, Any]:
        imid = _parse_image_id(image_id)
        gt_objects = self.get_ground_truth_objects(imid)
        units = self._caption_to_units(caption)

        object_mentions = []
        hallucinated_words = []
        real_words = []
        hallucination_idxs = []
        real_idxs = []
        recall_gt_objects = set()
        generated_node_words = []
        word_labels = []

        for unit in units:
            word = str(unit["word"])
            canonical = self.inverse_synonym_dict.get(word) if word in self.mscoco_object_set else None
            label = LABEL_IGNORE
            if canonical is not None:
                generated_node_words.append(canonical)
                label = LABEL_REAL if canonical in gt_objects else LABEL_HALLUCINATED
                mention = {
                    "surface": unit["surface"],
                    "surface_word": unit["surface"],
                    "word": word,
                    "normalized_word": word,
                    "canonical_name": canonical,
                    "canonical_object": canonical,
                    "word_idx": int(unit["word_idx"]),
                    "char_start": int(unit["char_start"]),
                    "char_end": int(unit["char_end"]),
                    "token_start": int(unit["word_idx"]),
                    "token_end": int(unit["word_idx"]) + int(unit["word_span"]),
                    "label": int(label),
                    "hallucinated": int(label == LABEL_HALLUCINATED),
                }
                object_mentions.append(mention)
                if label == LABEL_HALLUCINATED:
                    hallucinated_words.append((word, canonical))
                    hallucination_idxs.append(int(unit["word_idx"]))
                else:
                    real_words.append((word, canonical))
                    real_idxs.append(int(unit["word_idx"]))
                    recall_gt_objects.add(canonical)

            word_labels.append(
                {
                    "surface": unit["surface"],
                    "word": word,
                    "word_idx": int(unit["word_idx"]),
                    "char_start": int(unit["char_start"]),
                    "char_end": int(unit["char_end"]),
                    "canonical_object": canonical,
                    "label": int(label),
                }
            )

        num_objects = len(object_mentions)
        num_hallucinated = len(hallucinated_words)
        precision = (
            len(recall_gt_objects) / len(set(generated_node_words))
            if generated_node_words else 0.0
        )
        recall = len(recall_gt_objects) / len(gt_objects) if gt_objects else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        return {
            "image_id": imid,
            "caption": caption,
            "mscoco_hallucinated_words": hallucinated_words,
            "mscoco_real_words": real_words,
            "mscoco_gt_words": sorted(gt_objects),
            "mscoco_generated_words": list(generated_node_words),
            "hallucination_idxs": hallucination_idxs,
            "real_idxs": real_idxs,
            "words": [str(unit["word"]) for unit in units],
            "word_labels": word_labels,
            "object_mentions": object_mentions,
            "metrics": {
                "CHAIRs": int(num_hallucinated > 0),
                "CHAIRi": float(num_hallucinated / num_objects) if num_objects else 0.0,
                "Recall": float(recall),
                "Precision": float(precision),
                "F1": float(f1),
                "Len": float(0.01 * len(units)),
            },
        }

    def _caption_to_units(self, caption: str) -> list[dict[str, Any]]:
        lower_caption = caption.lower()
        spans = list(self._tokenizer.span_tokenize(lower_caption))
        words = [lower_caption[start:end] for start, end in spans]
        if not words:
            return []

        try:
            tagged_sent = nltk.pos_tag(words)
        except LookupError:
            ensure_nltk_data(download=False)
            tagged_sent = nltk.pos_tag(words)

        lemmas = []
        for word, tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag) or wordnet.NOUN
            lemmas.append(self._lemmatizer.lemmatize(word, pos=wordnet_pos))

        units = []
        i = 0
        while i < len(lemmas):
            double_word = " ".join(lemmas[i:i + 2])
            if double_word in self.double_word_dict and i + 1 < len(lemmas):
                char_start = int(spans[i][0])
                char_end = int(spans[i + 1][1])
                units.append(
                    {
                        "word": self.double_word_dict[double_word],
                        "surface": caption[char_start:char_end],
                        "word_idx": i,
                        "word_span": 2,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )
                i += 2
            else:
                char_start = int(spans[i][0])
                char_end = int(spans[i][1])
                units.append(
                    {
                        "word": lemmas[i],
                        "surface": caption[char_start:char_end],
                        "word_idx": i,
                        "word_span": 1,
                        "char_start": char_start,
                        "char_end": char_end,
                    }
                )
                i += 1

        if any(unit["word"] == "toilet" for unit in units) and any(
            unit["word"] == "seat" for unit in units
        ):
            units = [unit for unit in units if unit["word"] != "seat"]
        return units

    def _build_double_word_dict(self) -> dict[str, str]:
        double_word_dict = {_normalise_phrase(word): _normalise_phrase(word) for word in COCO_DOUBLE_WORDS}
        for word in ANIMAL_WORDS:
            double_word_dict[f"baby {word}"] = word
            double_word_dict[f"adult {word}"] = word
        for word in VEHICLE_WORDS:
            double_word_dict[f"passenger {word}"] = word
        double_word_dict["bow tie"] = "tie"
        double_word_dict["toilet seat"] = "toilet"
        double_word_dict["wine glas"] = "wine glass"
        return double_word_dict

    def _load_instances_payload(self) -> dict[str, Any]:
        if self.instances_file is not None:
            return _load_json(self.instances_file)
        if self.coco_path is None:
            raise ValueError("Either instances_file or coco_path is required.")
        return combine_coco_instances(self.coco_path)

    def _load_captions_payload(self) -> dict[str, Any] | None:
        if self.captions_file is not None:
            if not self.captions_file.exists():
                return None
            return _load_json(self.captions_file)
        if self.coco_path is None:
            return None
        return combine_coco_captions(self.coco_path)


def chair_summary(labeling: dict[str, dict] | dict[int, dict]) -> dict[str, float | int]:
    total_images = len(labeling)
    total_mentions = 0
    hallucinated_mentions = 0
    real_mentions = 0
    hallucinated_images = 0
    for item in labeling.values():
        spans = [
            span for span in item.get("object_token_spans", [])
            if span.get("label") in (LABEL_HALLUCINATED, LABEL_REAL)
        ]
        hall = sum(1 for span in spans if int(span.get("label")) == LABEL_HALLUCINATED)
        real = sum(1 for span in spans if int(span.get("label")) == LABEL_REAL)
        total_mentions += len(spans)
        hallucinated_mentions += hall
        real_mentions += real
        hallucinated_images += int(hall > 0)
    return {
        "images": total_images,
        "object_mentions": total_mentions,
        "hallucinated_mentions": hallucinated_mentions,
        "real_mentions": real_mentions,
        "images_with_hallucination": hallucinated_images,
        "chair_s": hallucinated_images / total_images if total_images else 0.0,
        "chair_i": hallucinated_mentions / total_mentions if total_mentions else 0.0,
        "label_semantics": "0=hallucinated, 1=real",
    }


def dedupe_mentions_by_object(mentions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first mention per canonical COCO object in caption order."""
    seen = set()
    result = []
    for mention in sorted(mentions, key=lambda item: int(item.get("char_start", 0))):
        key = str(mention.get("canonical_name") or mention.get("canonical_object") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(dict(mention))
    return result


def iter_ground_truth_entries(
    evaluator: CHAIR,
    image_ids: Iterable[int],
) -> Iterable[dict[str, Any]]:
    for image_id in image_ids:
        yield {
            "image_id": int(image_id),
            "image": f"COCO_val2014_{int(image_id):012d}.jpg",
            "objects": sorted(evaluator.get_ground_truth_objects(int(image_id))),
        }


def load_generated_captions(
    cap_file: str | Path,
    image_id_key: str,
    caption_key: str,
) -> tuple[list[str], list[int]]:
    cap_path = Path(cap_file)
    if cap_path.suffix == ".json":
        rows = _load_json(cap_path)
    elif cap_path.suffix == ".jsonl":
        rows = [json.loads(line) for line in cap_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported extension {cap_path.suffix} for cap_file={cap_file}")
    return [str(row[caption_key]) for row in rows], [int(row[image_id_key]) for row in rows]


def chair_eval(evaluator: CHAIR, image_id: int | str, caption: str) -> dict[str, Any]:
    return evaluator.compute_chair_token(image_id, caption)


CocoChairEvaluator = CHAIR


def _parse_synonyms(text: str) -> list[list[str]]:
    groups = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        groups.append([_normalise_phrase(part) for part in line.split(", ") if part.strip()])
    return groups


def _normalise_phrase(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_image_id(image_id: int | str) -> int:
    if isinstance(image_id, int):
        return image_id
    text = str(image_id)
    try:
        return int(text)
    except ValueError:
        pass
    match = re.search(r"_(\d+)\.jpg$", text)
    if match:
        number = match.group(1)
        return int(number.lstrip("0") or "0")
    raise ValueError(f"Cannot parse COCO image_id from {image_id!r}")


def _cache_is_compatible(obj: Any) -> bool:
    return (
        getattr(obj, "cache_version", None) == CHAIR_CACHE_VERSION
        and hasattr(obj, "inverse_synonym_dict")
        and hasattr(obj, "double_word_dict")
        and hasattr(obj, "imid_to_objects")
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
