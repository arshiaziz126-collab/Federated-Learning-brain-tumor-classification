"""Inspect the dataset in data/brain_tumor_dataset/ and print a report.

    python inspect_dataset.py
    python inspect_dataset.py --data-dir path/to/dataset

The report is also saved to outputs/dataset_report.json (and reused by the app).
"""
import argparse

import config
from backend.dataset import DatasetError, inspect_dataset
from backend.io_utils import format_bytes, save_json


def print_report(rep: dict) -> None:
    line = "-" * 64
    print(line)
    print(f"Dataset root      : {rep['dataset_root']}")
    print(f"Structure         : {rep['structure']}")
    print(f"Images scanned    : {rep['total_files_scanned']}  (valid {rep['valid_images']}, corrupted {rep['corrupted_images']})")
    print(f"Total size        : {format_bytes(rep['total_bytes'])}")
    print(f"Formats           : {rep['image_formats']}")
    print(f"Colour modes      : {rep['color_modes']}")
    d = rep.get("dimensions") or {}
    if d:
        print(f"Dimensions        : width {d['min_width']}-{d['max_width']}, height {d['min_height']}-{d['max_height']}, "
              f"{d['unique_sizes']} distinct sizes; most common: "
              + ", ".join(f"{x['size']} (x{x['count']})" for x in d["most_common"][:3]))
    print(f"Classes ({rep['num_classes']})     : {rep['class_names']}")
    print("Class distribution:")
    for c in rep["class_names"]:
        print(f"    {c:<22}{rep['class_counts'].get(c, 0):>7}")
    if rep["class_imbalance_ratio"]:
        print(f"Imbalance ratio   : {rep['class_imbalance_ratio']:.2f} (largest class / smallest class)")
    for s, counts in rep["split_class_counts"].items():
        print(f"Split '{s}'        : {sum(counts.values())} images  {counts}")
    dup = rep["duplicates"]
    print(f"Exact duplicates  : {dup['redundant_copies']} redundant copies in {dup['duplicate_groups']} groups; "
          f"{dup['groups_spanning_splits']} groups span train/test; "
          f"{dup['groups_with_conflicting_labels']} groups have conflicting labels")
    if rep["unlabeled_images"]:
        print(f"Unlabeled images  : {rep['unlabeled_images']} (not inside a class folder; ignored)")
    if rep["non_image_files_by_extension"]:
        print(f"Other files       : {rep['non_image_files_by_extension']}")
    for c in rep["corrupted_examples"][:5]:
        print(f"    corrupted: {c['file']} ({c['error']})")
    print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None, help=f"default: {config.DATA_DIR}")
    args = ap.parse_args()
    try:
        report, _ = inspect_dataset(args.data_dir)
    except DatasetError as e:
        raise SystemExit(f"\nERROR: {e}")
    print_report(report)
    save_json(report, config.OUT["dataset_report"])
    print(f"Saved {config.OUT['dataset_report'].relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
