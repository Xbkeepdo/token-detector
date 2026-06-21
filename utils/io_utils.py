"""Pickle and JSON I/O helpers."""

import json
import os
import pickle
import tempfile


def save_pkl(obj, path):
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".pkl", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def append_pkl(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "ab") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    objects = []
    with open(path, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
            except pickle.UnpicklingError:
                if objects:
                    break
                raise
    if not objects:
        return []
    if len(objects) == 1:
        return objects[0]

    merged = []
    for obj in objects:
        if isinstance(obj, list):
            merged.extend(obj)
        else:
            merged.append(obj)
    return merged


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
