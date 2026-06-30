"""GPT-4o based labeling of hallucinated object tokens in generated captions."""

from __future__ import annotations
import json
import os
import pickle
import importlib.util
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from openai import OpenAI

from labeling.token_finder import find_object_token_spans
from utils.io_utils import save_json, load_json


SYSTEM_PROMPT = (
    "You are a precise hallucination detector. "
    "Follow the instructions exactly and output ONLY a JSON list."
)

DEFAULT_OPENAI_MODEL = "gpt-4o"
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
GITHUB_MODELS_MODEL = "openai/gpt-4o"

USER_TEMPLATE = """\
You are given:
- A list of ground truth object classes (from COCO).
- A detailed description of an image.
- Several captions of the same image.

Your task:
Find all object classes that are mentioned in the description, but are NOT mentioned in any of the captions, and are NOT present in the ground truth list.

Output the result as a list as in the examples. Do NOT add any extra text or provide any explanations.

Examples:
Objects: ["bowl", "broccoli", "carrot"]
Description: There are two bowls of food, one containing a mix of vegetables, such as broccoli and carrots, and the other containing meat. Captions:
- A bowl with broccoli and carrots.
→ Output: ["meat"]

Objects: ["bowl", "broccoli"]
Description: - A bowl full of broccoli.
Captions: - A bowl of green vegetables.
→ Output: []

Now answer:
Objects: {objects}
Description: {description}
Captions:
{captions_formatted}
→ Output:"""


def _format_captions(captions: List[str]) -> str:
    """Format captions as a bullet list, matching the paper's template."""
    return "\n".join(f"- {cap.strip()}" for cap in captions)


def _call_gpt4o(
    client: OpenAI,
    objects: List[str],
    description: str,
    captions: List[str],
    model: str = DEFAULT_OPENAI_MODEL,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> Optional[List[str]]:
    """
    Call GPT-4o with the paper's exact prompt and parse the output as a
    Python list of hallucinated object class strings.

    Returns None on API failure so the sample can be retried with --resume.
    """
    captions_formatted = _format_captions(captions)
    user_msg = USER_TEMPLATE.format(
        objects=json.dumps(objects),
        description=description,
        captions_formatted=captions_formatted,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content.strip()

            raw = raw.replace("```json", "").replace("```", "").strip()
            hallucinated = json.loads(raw)
            if isinstance(hallucinated, list):
                return [str(w).lower() for w in hallucinated]
            return []

        except json.JSONDecodeError:
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[GPT4Labeler] Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay * (attempt + 1))
            else:
                print(f"[GPT4Labeler] Failed after {max_retries} retries: {e}")
                return None


def _is_github_models_base_url(base_url: Optional[str]) -> bool:
    return bool(base_url and "models.github.ai" in base_url)


def _resolve_api_settings(
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    openai_model: Optional[str],
) -> tuple[str, Optional[str], str]:
    env_openai_key = os.environ.get("OPENAI_API_KEY")
    github_models_token = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    base_url = openai_base_url or os.environ.get("OPENAI_BASE_URL")

    if not base_url and not (openai_api_key or env_openai_key) and github_models_token:
        base_url = GITHUB_MODELS_BASE_URL

    if _is_github_models_base_url(base_url):
        api_key = openai_api_key or github_models_token or env_openai_key or ""
    else:
        api_key = openai_api_key or env_openai_key or github_models_token or ""

    model = openai_model or os.environ.get("OPENAI_MODEL")
    if not model:
        model = GITHUB_MODELS_MODEL if _is_github_models_base_url(base_url) else DEFAULT_OPENAI_MODEL

    return api_key, base_url, model


def _get_coco_objects(sample: dict) -> List[str]:
    """Return unique COCO object names from either new or legacy sample format."""
    if "coco_objects" in sample:
        return list(sample["coco_objects"])

    return sorted({
        ann["category_name"]
        for ann in sample.get("annotations", [])
        if ann.get("category_name")
    })


def _load_chair_cache_objects(cache_path: Optional[str]) -> Optional[dict[int, set[str]]]:
    if not cache_path:
        return None
    _ensure_coco_chair_module()
    with open(cache_path, "rb") as handle:
        cache_obj = pickle.load(handle)
    image_id_to_objects = getattr(cache_obj, "image_id_to_objects", None)
    if image_id_to_objects is None and isinstance(cache_obj, dict):
        image_id_to_objects = cache_obj.get("image_id_to_objects", cache_obj)
    if image_id_to_objects is None:
        raise ValueError(f"CHAIR cache at {cache_path!r} has no image_id_to_objects mapping.")
    result = {
        int(image_id): {str(item).lower() for item in objects}
        for image_id, objects in image_id_to_objects.items()
    }
    print(f"[GPT4Labeler] Loaded CHAIR cache objects for {len(result)} images: {cache_path}")
    return result


def _ensure_coco_chair_module() -> None:
    if "coco_chair" in sys.modules:
        return
    repo_root = Path(__file__).resolve().parents[1]
    chair_path = repo_root / "coco-labeling" / "coco_chair.py"
    if not chair_path.exists():
        return
    spec = importlib.util.spec_from_file_location("coco_chair", chair_path)
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["coco_chair"] = module
    spec.loader.exec_module(module)


def _merge_real_objects(coco_objects: List[str], chair_objects: Optional[set[str]]) -> List[str]:
    merged = {str(item).lower() for item in coco_objects}
    if chair_objects:
        merged.update(str(item).lower() for item in chair_objects)
    return sorted(merged)


def label_dataset(
    samples: List[dict],
    generation_results: Dict[int, str],   # image_id → generated_text
    generation_token_ids: Dict[int, List[int]],  # image_id → response_token_ids
    tokenizer,                            # the model's tokenizer (for token finding)
    output_path: str,
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_proxy: Optional[str] = None,
    chair_cache_path: Optional[str] = None,
    resume: bool = True,
    sleep_between_calls: float = 0.5,
) -> Dict[int, dict]:
    """
    Run GPT-4o labeling on all samples and save results.

    Args:
        samples:               COCO sample list (from coco_loader).
        generation_results:    Dict mapping image_id → generated description string.
        generation_token_ids:  Dict mapping image_id → list of response token ids.
        tokenizer:             The LVLM tokenizer (used for token-position finding).
        output_path:           Path to save/resume the JSON results file.
        openai_api_key:        API key (falls back to OPENAI_API_KEY, then
                               GITHUB_MODELS_TOKEN/GITHUB_TOKEN env vars).
        openai_base_url:       Optional OpenAI-compatible API base URL.
        openai_model:          Chat model name, e.g. gpt-4o or openai/gpt-4o.
        openai_proxy:          Optional HTTP(S) proxy URL for OpenAI requests.
        chair_cache_path:      Optional coco_chair_gt.pkl used to expand real objects.
        resume:                Skip already-labeled images if output_path exists.
        sleep_between_calls:   Seconds to sleep between API calls.

    Returns:
        Dict mapping image_id → labeling result dict.
    """
    api_key, base_url, model = _resolve_api_settings(
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
    )
    proxy = openai_proxy or os.environ.get("OPENAI_PROXY")
    http_client = httpx.Client(proxy=proxy, timeout=60.0) if proxy else None

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if http_client:
        client_kwargs["http_client"] = http_client
    if _is_github_models_base_url(base_url):
        client_kwargs["default_headers"] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    client = OpenAI(**client_kwargs)
    chair_cache_objects = _load_chair_cache_objects(chair_cache_path)

    results: Dict[int, dict] = {}
    if resume and os.path.exists(output_path):
        raw = load_json(output_path)
        results = {int(k): v for k, v in raw.items()}
        print(f"[GPT4Labeler] Resuming — {len(results)} images already labeled.")

    for sample in samples:
        image_id = sample["image_id"]
        if image_id in results:
            continue
        if image_id not in generation_results:
            continue

        generated_text = generation_results[image_id]
        coco_objects = _get_coco_objects(sample)    # list[str], ground-truth classes
        real_objects = _merge_real_objects(
            coco_objects,
            chair_cache_objects.get(int(image_id)) if chair_cache_objects is not None else None,
        )
        captions = sample["captions"]               # list[str], GT captions

        hallucinated_words = _call_gpt4o(
            client,
            objects=real_objects,
            description=generated_text,
            captions=captions,
            model=model,
        )
        if hallucinated_words is None:
            print(f"[GPT4Labeler] Skipping image {image_id}; will retry on resume.")
            continue
        real_object_set = {item.lower() for item in real_objects}
        hallucinated_words = [
            word for word in hallucinated_words
            if word.lower() not in real_object_set
        ]

        time.sleep(sleep_between_calls)

        response_token_ids = generation_token_ids.get(image_id, [])
        object_token_spans = find_object_token_spans(
            generated_text=generated_text,
            response_token_ids=response_token_ids,
            hallucinated_words=hallucinated_words,
            coco_objects=real_objects,
            tokenizer=tokenizer,
        )

        results[image_id] = {
            "image_id": image_id,
            "generated_text": generated_text,
            "hallucinated_words": hallucinated_words,
            "object_token_spans": object_token_spans,
        }

        save_json({str(k): v for k, v in results.items()}, output_path)

    print(f"[GPT4Labeler] Done. {len(results)} images labeled.")
    return results
