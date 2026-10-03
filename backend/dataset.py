"""Dataset discovery, inspection and federated partitioning.

Nothing here assumes a particular dataset layout or class list:

* classes are the folder names found on disk,
* a pre-made train/test split is used only if the folders exist,
* every statistic in the report is computed from the files themselves.
"""
from __future__ import annotations

import hashlib
import io
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

import config
from backend.preprocessing import load_images


class DatasetError(RuntimeError):
    """Raised when the dataset folder is missing or unusable."""


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
def find_dataset_root(data_dir=None) -> Path:
    """Return the folder that directly contains the class (or split) folders.

    Kaggle zips often wrap everything in one extra folder (e.g. ``archive/``);
    single wrapper folders are descended automatically.
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    if not data_dir.exists():
        raise DatasetError(
            f"Dataset folder not found: {data_dir}\n"
            f"Download the Kaggle brain-tumour MRI dataset and extract it into "
            f"'{config.DATA_DIR.relative_to(config.ROOT)}'."
        )
    root = data_dir
    for _ in range(4):
        entries = [p for p in root.iterdir() if not p.name.startswith(".")]
        has_images = any(p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS for p in entries)
        subdirs = [p for p in entries if p.is_dir()]
        if not has_images and len(subdirs) == 1:
            root = subdirs[0]
        else:
            break
    return root


def _split_and_label(rel_parts):
    """Derive (split, class label) from the folder parts of an image path."""
    dirs = list(rel_parts[:-1])
    for i, part in enumerate(dirs):
        key = part.lower()
        if key in config.SPLIT_ALIASES:
            split = config.SPLIT_ALIASES[key]
            if i + 1 < len(dirs):
                return split, dirs[i + 1]
            if i > 0:                       # layout: <class>/<split>/img
                return split, dirs[0]
            return split, None
    return None, (dirs[0] if dirs else None)


def _inspect_file(path: Path, root: Path) -> dict:
    rel = path.relative_to(root)
    split, label = _split_and_label(rel.parts)
    rec = {
        "path": str(path), "rel": rel.as_posix(), "split": split, "label": label,
        "md5": None, "width": None, "height": None, "mode": None,
        "bytes": None, "valid": False, "error": None,
    }
    try:
        data = path.read_bytes()
        rec["bytes"] = len(data)
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.load()                        # forces a full decode -> catches truncated files
            rec["width"], rec["height"] = im.size
            rec["mode"] = im.mode
        rec["valid"] = True
    except Exception as exc:                 # corrupted / unreadable image
        rec["error"] = f"{type(exc).__name__}: {exc}"
    return rec


# ---------------------------------------------------------------------------
# Inspection report
# ---------------------------------------------------------------------------
def inspect_dataset(data_dir=None, progress: bool = True):
    """Scan the dataset and return ``(report, records)``.

    ``records`` has one dict per image file; ``report`` is JSON-serialisable.
    """
    root = find_dataset_root(data_dir)
    image_files, other_ext = [], Counter()
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if p.suffix.lower() in config.IMAGE_EXTENSIONS:
            image_files.append(p)
        else:
            other_ext[p.suffix.lower() or "(no extension)"] += 1
    if not image_files:
        raise DatasetError(f"No image files ({', '.join(config.IMAGE_EXTENSIONS)}) found under {root}")

    records = []
    for i, p in enumerate(image_files, 1):
        records.append(_inspect_file(p, root))
        if progress and (i % 1000 == 0 or i == len(image_files)):
            print(f"  scanned {i}/{len(image_files)} images", flush=True)

    return _build_report(root, records, other_ext), records


def _build_report(root: Path, records: list, other_ext: Counter) -> dict:
    valid = [r for r in records if r["valid"]]
    corrupted = [r for r in records if not r["valid"]]
    unlabeled = [r for r in valid if not r["label"]]
    labeled = [r for r in valid if r["label"]]

    class_names = sorted({r["label"] for r in labeled})
    class_counts = Counter(r["label"] for r in labeled)
    splits = sorted({r["split"] for r in labeled if r["split"]})
    split_class_counts = {
        s: dict(Counter(r["label"] for r in labeled if r["split"] == s)) for s in splits
    }
    if splits:
        structure = "pre-split folders (" + ", ".join(splits) + ") containing class sub-folders"
    else:
        structure = "class sub-folders only (no train/test split in the dataset)"

    # duplicates (exact byte-level, by MD5)
    groups = defaultdict(list)
    for r in valid:
        groups[r["md5"]].append(r)
    dup_groups = [g for g in groups.values() if len(g) > 1]
    cross_split = [g for g in dup_groups if len({x["split"] for x in g}) > 1]
    conflicts = [g for g in dup_groups if len({x["label"] for x in g}) > 1]

    sizes = Counter(f"{r['width']}x{r['height']}" for r in valid)
    widths = [r["width"] for r in valid]
    heights = [r["height"] for r in valid]
    counts = list(class_counts.values()) or [0]

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "structure": structure,
        "has_predefined_test_split": "test" in splits,
        "splits_found": splits,
        "total_files_scanned": len(records),
        "valid_images": len(valid),
        "corrupted_images": len(corrupted),
        "corrupted_examples": [{"file": r["rel"], "error": r["error"]} for r in corrupted[:20]],
        "unlabeled_images": len(unlabeled),
        "non_image_files_by_extension": dict(other_ext),
        "num_classes": len(class_names),
        "class_names": class_names,
        "class_counts": dict(class_counts),
        "split_class_counts": split_class_counts,
        "class_imbalance_ratio": (max(counts) / min(counts)) if min(counts) else None,
        "image_formats": dict(Counter(Path(r["rel"]).suffix.lower() for r in valid)),
        "color_modes": dict(Counter(r["mode"] for r in valid)),
        "dimensions": {
            "min_width": int(min(widths)), "max_width": int(max(widths)),
            "min_height": int(min(heights)), "max_height": int(max(heights)),
            "mean_width": float(np.mean(widths)), "mean_height": float(np.mean(heights)),
            "unique_sizes": len(sizes),
            "most_common": [{"size": s, "count": c} for s, c in sizes.most_common(5)],
        } if valid else {},
        "total_bytes": int(sum(r["bytes"] or 0 for r in valid)),
        "duplicates": {
            "duplicate_groups": len(dup_groups),
            "redundant_copies": int(sum(len(g) - 1 for g in dup_groups)),
            "groups_spanning_splits": len(cross_split),
            "groups_with_conflicting_labels": len(conflicts),
            "examples": [[x["rel"] for x in g[:4]] for g in dup_groups[:5]],
            "method": "exact byte-identical files (MD5); near-duplicates are not detected",
        },
    }


# ---------------------------------------------------------------------------
# Cleaning + partitioning
# ---------------------------------------------------------------------------
def _dedupe(records: list, enabled: bool):
    """Remove exact duplicates and images whose duplicates carry different labels.

    Priority when duplicates exist: test > val > train, so a training image that
    also appears in the test set is dropped (prevents train/test leakage).
    """
    stats = {"enabled": enabled, "dropped_exact_duplicates": 0,
             "dropped_conflicting_label_images": 0, "dropped_train_copies_of_test_images": 0}
    if not enabled:
        return records, stats
    prio = {"test": 0, "val": 1, "train": 2, None: 3}
    groups = defaultdict(list)
    for r in records:
        groups[r["md5"]].append(r)
    keep = []
    for g in groups.values():
        if len(g) == 1:
            keep.append(g[0])
            continue
        if len({x["label"] for x in g}) > 1:
            stats["dropped_conflicting_label_images"] += len(g)
            continue
        g = sorted(g, key=lambda x: (prio[x["split"]], x["rel"]))
        keep.append(g[0])
        stats["dropped_exact_duplicates"] += len(g) - 1
        if g[0]["split"] == "test":
            stats["dropped_train_copies_of_test_images"] += sum(1 for x in g[1:] if x["split"] != "test")
    return sorted(keep, key=lambda r: r["rel"]), stats


def _stratified_split(recs: list, fraction: float, rng) -> tuple:
    """Return (kept, held_out) with ``fraction`` of every class held out."""
    by_class = defaultdict(list)
    for r in recs:
        by_class[r["label"]].append(r)
    kept, held = [], []
    for label in sorted(by_class):
        items = by_class[label]
        order = rng.permutation(len(items))
        n_hold = int(round(fraction * len(items))) if len(items) >= 5 else 0
        held.extend(items[i] for i in order[:n_hold])
        kept.extend(items[i] for i in order[n_hold:])
    return kept, held


def _iid_partition(labels: np.ndarray, n_clients: int, rng):
    parts = [[] for _ in range(n_clients)]
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_clients)):
            parts[k].extend(chunk.tolist())
    return parts, 1


def _dirichlet_partition(labels: np.ndarray, n_clients: int, alpha: float, rng):
    """Label-skewed (non-IID) split: per class, shares ~ Dirichlet(alpha).

    Retries (deterministically, same RNG stream) until every hospital has a
    reasonable amount of data and at least two classes.
    """
    classes = np.unique(labels)
    min_size = int(0.5 * len(labels) / n_clients)
    best, best_score = None, -1
    for attempt in range(1, 501):
        parts = [[] for _ in range(n_clients)]
        for c in classes:
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet([alpha] * n_clients)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                parts[k].extend(chunk.tolist())
        sizes = [len(p) for p in parts]
        n_cls = [len({int(labels[i]) for i in p}) for p in parts]
        score = min(sizes)
        if min(sizes) >= min_size and min(n_cls) >= min(2, len(classes)):
            return parts, attempt
        if score > best_score:
            best, best_score = parts, score
    print("  warning: could not satisfy the minimum-size rule; using the most balanced attempt")
    return best, 500


def prepare_federated_data(records: list, *, num_clients: int, partition: str, alpha: float,
                           val_fraction: float, test_fraction: float, seed: int,
                           dedupe: bool, dataset_root: str) -> dict:
    """Clean, split and physically partition the dataset into hospital folders.

    Returns the partition manifest (also written by the caller to outputs/).
    """
    rng = np.random.default_rng(seed)
    usable = sorted([r for r in records if r["valid"] and r["label"]], key=lambda r: r["rel"])
    usable, dedupe_stats = _dedupe(usable, dedupe)
    class_names = sorted({r["label"] for r in usable})
    if len(class_names) < 2:
        raise DatasetError(f"Need at least 2 classes, found {len(class_names)}: {class_names}")
    class_index = {c: i for i, c in enumerate(class_names)}

    # 1) global held-out test set (never used for training or model selection)
    test_recs = [r for r in usable if r["split"] == "test"]
    pool = [r for r in usable if r["split"] != "test"]
    if test_recs:
        test_source = "test folder that ships with the dataset"
    else:
        pool, test_recs = _stratified_split(pool, test_fraction, rng)
        test_source = f"random stratified {test_fraction:.0%} hold-out (dataset has no test folder), seed {seed}"

    # 2) split the training pool between hospitals
    labels = np.array([class_index[r["label"]] for r in pool])
    if partition == "iid":
        parts, attempts = _iid_partition(labels, num_clients, rng)
    elif partition == "dirichlet":
        parts, attempts = _dirichlet_partition(labels, num_clients, alpha, rng)
    else:
        raise ValueError(f"Unknown partition mode: {partition}")

    # 3) materialise isolated hospital folders + local train/val split
    if config.CLIENTS_DIR.exists():
        shutil.rmtree(config.CLIENTS_DIR)
    clients = []
    for k in range(num_clients):
        recs = sorted([pool[i] for i in parts[k]], key=lambda r: r["rel"])
        local_train, local_val = _stratified_split(recs, val_fraction, rng)
        cid = config.client_id(k)
        cdir = config.CLIENTS_DIR / cid
        entry = {"client_id": cid, "name": config.client_name(k),
                 "dir": cdir.relative_to(config.ROOT).as_posix(), "raw_bytes": 0}
        for split_name, items in (("train", local_train), ("val", local_val)):
            files = []
            for j, r in enumerate(items):
                dest_rel = f"{r['label']}/{split_name}_{j:05d}_{Path(r['path']).name}"
                dest = cdir / dest_rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r["path"], dest)
                files.append({"file": dest_rel, "label": r["label"], "md5": r["md5"], "bytes": r["bytes"]})
                entry["raw_bytes"] += r["bytes"]
            entry[split_name] = {
                "count": len(files),
                "class_counts": {c: sum(1 for f in files if f["label"] == c) for c in class_names},
                "files": files,
            }
        clients.append(entry)

    test = {
        "root": dataset_root, "source": test_source, "count": len(test_recs),
        "class_counts": {c: sum(1 for r in test_recs if r["label"] == c) for c in class_names},
        "files": [{"file": r["rel"], "label": r["label"], "md5": r["md5"], "bytes": r["bytes"]} for r in test_recs],
    }
    return {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "partition": {"mode": partition, "alpha": alpha if partition == "dirichlet" else None, "attempts": attempts},
        "val_fraction": val_fraction,
        "class_names": class_names,
        "cleaning": dedupe_stats,
        "images_after_cleaning": len(usable),
        "test": test,
        "clients": clients,
    }


# ---------------------------------------------------------------------------
# Loading arrays for training / evaluation
# ---------------------------------------------------------------------------
def load_client_arrays(entry: dict, split: str, class_names: list, img_size: int, grayscale: bool):
    """Load one hospital's images from *its own folder only*."""
    class_index = {c: i for i, c in enumerate(class_names)}
    cdir = config.ROOT / entry["dir"]
    files = entry[split]["files"]
    X = load_images([cdir / f["file"] for f in files], img_size, grayscale)
    y = np.array([class_index[f["label"]] for f in files], dtype=np.int64)
    return X, y


def load_test_arrays(manifest: dict, img_size: int, grayscale: bool):
    class_index = {c: i for i, c in enumerate(manifest["class_names"])}
    root = Path(manifest["test"]["root"])
    files = manifest["test"]["files"]
    X = load_images([root / f["file"] for f in files], img_size, grayscale)
    y = np.array([class_index[f["label"]] for f in files], dtype=np.int64)
    return X, y
