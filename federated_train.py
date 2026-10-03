"""Train NeuroFed AI: federated learning across simulated hospitals + a centralized baseline.

    python federated_train.py                       # sensible defaults
    python federated_train.py --rounds 20 --local-epochs 2
    python federated_train.py --partition iid       # compare with an IID split
    python federated_train.py --skip-centralized    # federated only
    python federated_train.py --algorithm fedprox --mu 0.01 --skip-centralized --tag v2_fedprox

--tag NAME also copies outputs/ and models/ to results_NAME/ when training ends, so runs
do not overwrite each other.

Everything the Streamlit app displays is written to outputs/ and models/ by this script.
The app itself never trains anything.
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config
from backend.dataset import DatasetError, inspect_dataset, load_test_arrays, prepare_federated_data
from backend.io_utils import format_bytes, save_json, set_global_seed
from inspect_dataset import print_report

D = config.DEFAULTS


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=None, help=f"dataset folder (default {config.DATA_DIR.relative_to(config.ROOT)})")
    ap.add_argument("--num-clients", type=int, default=D["num_clients"], help="number of hospitals")
    ap.add_argument("--rounds", type=int, default=D["rounds"], help="federated communication rounds")
    ap.add_argument("--local-epochs", type=int, default=D["local_epochs"], help="epochs each hospital trains per round")
    ap.add_argument("--batch-size", type=int, default=D["batch_size"])
    ap.add_argument("--lr", type=float, default=D["learning_rate"], help="learning rate (Adam)")
    ap.add_argument("--img-size", type=int, default=D["img_size"], help="images are resized to N x N")
    ap.add_argument("--rgb", action="store_true", help="use 3 colour channels instead of grayscale")
    ap.add_argument("--arch", default=D["arch"], help="architecture name from backend/model.py")
    ap.add_argument("--partition", choices=["dirichlet", "iid"], default=D["partition"],
                    help="how the data is split between hospitals")
    ap.add_argument("--alpha", type=float, default=D["alpha"], help="Dirichlet alpha (smaller = more non-IID)")
    ap.add_argument("--val-fraction", type=float, default=D["val_fraction"], help="local validation share per hospital")
    ap.add_argument("--test-fraction", type=float, default=D["test_fraction"],
                    help="hold-out share, used only if the dataset has no test folder")
    ap.add_argument("--client-fraction", type=float, default=D["client_fraction"], help="share of hospitals per round")
    ap.add_argument("--centralized-epochs", type=int, default=None,
                    help="epochs for the centralized baseline (default: rounds x local-epochs)")
    ap.add_argument("--skip-centralized", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true", help="keep exact duplicate images")
    ap.add_argument("--seed", type=int, default=D["seed"])
    ap.add_argument("--algorithm", choices=["fedavg", "fedprox"], default=D["algorithm"])
    ap.add_argument("--mu", type=float, default=D["mu"], help="FedProx proximal strength")
    ap.add_argument("--tag", default=None, help="copy outputs/ and models/ to results_<tag>/ at the end")
    return ap.parse_args()


def _dist_table(manifest):
    names = manifest["class_names"]
    w = max(len(c) for c in names + ["Hospital A"]) + 2
    print("  " + "".ljust(w) + "".join(f"{c[:12]:>14}" for c in names) + f"{'total':>10}")
    for c in manifest["clients"]:
        row = c["train"]["class_counts"]
        print("  " + c["name"].ljust(w) + "".join(f"{row[k]:>14}" for k in names) + f"{c['train']['count']:>10}   (+{c['val']['count']} local val)")
    t = manifest["test"]
    print("  " + "Held-out test".ljust(w) + "".join(f"{t['class_counts'][k]:>14}" for k in names) + f"{t['count']:>10}")


def main():
    args = parse_args()
    set_global_seed(args.seed)
    mu = args.mu if args.algorithm == "fedprox" else 0.0
    print(f"Algorithm: {args.algorithm}"
          + (f" (mu={mu})" if mu else ""))
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.OUTPUTS_DIR.mkdir(exist_ok=True)

    # ---- 1. inspect --------------------------------------------------------------------------
    print("\n[1/6] Inspecting dataset")
    try:
        report, records = inspect_dataset(args.data_dir)
    except DatasetError as e:
        sys.exit(f"\nERROR: {e}")
    print_report(report)
    save_json(report, config.OUT["dataset_report"])

    # ---- 2. partition between hospitals -------------------------------------------------------
    print("\n[2/6] Partitioning data between hospitals "
          f"({args.partition}{f', alpha={args.alpha}' if args.partition == 'dirichlet' else ''}, seed={args.seed})")
    try:
        manifest = prepare_federated_data(
            records, num_clients=args.num_clients, partition=args.partition, alpha=args.alpha,
            val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=args.seed,
            dedupe=not args.no_dedupe, dataset_root=report["dataset_root"])
    except DatasetError as e:
        sys.exit(f"\nERROR: {e}")
    save_json(manifest, config.OUT["partition"])
    cl = manifest["cleaning"]
    if cl["enabled"]:
        print(f"  cleaning: dropped {cl['dropped_exact_duplicates']} exact duplicates and "
              f"{cl['dropped_conflicting_label_images']} label-conflicting images "
              f"({cl['dropped_train_copies_of_test_images']} were training copies of test images)")
    print(f"  test set: {manifest['test']['source']}")
    _dist_table(manifest)

    # heavy imports only after the dataset is confirmed usable
    from backend import communication
    from backend.centralized import train_centralized
    from backend.evaluation import evaluate_model, wilson_interval
    from backend.federated import run_federated_training
    from backend.local_training import HospitalClient
    from backend.model import architecture_summary, build_model, count_parameters, get_parameters, set_parameters
    from backend.privacy import build_privacy_report
    import keras
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    class_names = manifest["class_names"]
    channels = 3 if args.rgb else 1
    model_cfg = {"arch": args.arch, "img_size": args.img_size, "grayscale": not args.rgb,
                 "input_shape": [args.img_size, args.img_size, channels], "class_names": class_names}

    # ---- 3. load hospital data locally --------------------------------------------------------
    print("\n[3/6] Loading each hospital's images from its own folder")
    clients = [HospitalClient(e, class_names, model_cfg, args.seed) for e in manifest["clients"]]
    for c in clients:
        print(f"  {c.name}: {c.num_train} train / {c.num_val} local-val images")
    test_data = load_test_arrays(manifest, args.img_size, not args.rgb)
    print(f"  held-out test set: {len(test_data[1])} images")

    # ---- 4. federated training ----------------------------------------------------------------
    print(f"\n[4/6] Federated training ({args.algorithm}): {args.rounds} rounds x {args.local_epochs} local epochs, "
          f"{len(clients)} hospitals")
    server_model = build_model(args.arch, tuple(model_cfg["input_shape"]), len(class_names))
    n_params = count_parameters(server_model)
    print(f"  model: {n_params:,} trainable parameters")
    tracker = communication.CommunicationTracker()
    history, best_params, best_round, stats = run_federated_training(
        clients, server_model, class_names, test_data, rounds=args.rounds, local_epochs=args.local_epochs,
        batch_size=args.batch_size, learning_rate=args.lr, client_fraction=args.client_fraction,
        seed=args.seed, tracker=tracker, mu=mu)
    set_parameters(server_model, best_params)
    server_model.save(config.GLOBAL_MODEL_PATH)
    fed_test = evaluate_model(server_model, *test_data, class_names)
    lo, hi = wilson_interval(round(fed_test["accuracy"] * fed_test["num_samples"]), fed_test["num_samples"])
    print(f"  saved {config.GLOBAL_MODEL_PATH.relative_to(config.ROOT)} (round {best_round}, chosen on federated validation)")
    print(f"  held-out test: accuracy {fed_test['accuracy']:.4f}  macro-F1 {fed_test['f1_macro']:.4f}")

    # ---- 5. centralized baseline ---------------------------------------------------------------
    central = None
    if not args.skip_centralized:
        epochs = args.centralized_epochs or args.rounds * args.local_epochs
        print(f"\n[5/6] Centralized baseline: all data pooled, {epochs} epochs")
        Xtr = np.concatenate([c.train_arrays()[0] for c in clients])
        ytr = np.concatenate([c.train_arrays()[1] for c in clients])
        Xva = np.concatenate([c.val_arrays()[0] for c in clients])
        yva = np.concatenate([c.val_arrays()[1] for c in clients])
        c_model, c_hist, c_best, c_test, c_secs = train_centralized(
            Xtr, ytr, Xva, yva, test_data, class_names, arch=args.arch, input_shape=tuple(model_cfg["input_shape"]),
            epochs=epochs, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)
        c_model.save(config.CENTRALIZED_MODEL_PATH)
        clo, chi = wilson_interval(round(c_test["accuracy"] * c_test["num_samples"]), c_test["num_samples"])
        central = {"epochs": epochs, "best_epoch": c_best, "selection_criterion": "highest validation accuracy (pooled validation data)",
                   "num_train": int(len(ytr)), "num_val": int(len(yva)), "training_seconds": c_secs,
                   "test": c_test, "test_accuracy_ci95": [clo, chi], "history": c_hist}
        save_json(central, config.OUT["centralized"])
        del Xtr, Xva
        print(f"  held-out test: accuracy {c_test['accuracy']:.4f}  macro-F1 {c_test['f1_macro']:.4f}")
    else:
        print("\n[5/6] Centralized baseline skipped")

    # ---- 6. summaries --------------------------------------------------------------------------
    print("\n[6/6] Writing results")
    raw_by_client = {c["client_id"]: c["raw_bytes"] for c in manifest["clients"]}
    payload_bytes = len(communication.pack(get_parameters(server_model)))
    comm = tracker.summary(num_clients=len(clients), raw_bytes_by_client=raw_by_client,
                           model_parameters=n_params, model_payload_bytes=payload_bytes)
    save_json(comm, config.OUT["communication"])

    shapes = [tuple(p.shape) for p in get_parameters(server_model)]
    privacy = build_privacy_report(manifest, comm, len(shapes), shapes, stats["updates_validated"])
    save_json(privacy, config.OUT["privacy"])

    best_rec = history[best_round - 1]
    client_records = []
    for entry, c in zip(manifest["clients"], clients):
        rows = [r for h in history for r in h["clients"] if r["client_id"] == c.client_id]
        part = [r for r in rows if r["participated"]]
        at_best = next(r for r in best_rec["clients"] if r["client_id"] == c.client_id)
        client_records.append({
            "client_id": c.client_id, "name": c.name, "data_dir": entry["dir"],
            "num_train": c.num_train, "num_val": c.num_val,
            "class_counts_train": c.class_counts("train"), "class_counts_val": c.class_counts("val"),
            "raw_data_bytes": entry["raw_bytes"],
            "rounds_participated": len(part), "rounds_total": len(history),
            "status": "Participated in every round" if len(part) == len(history) else f"Participated in {len(part)} of {len(history)} rounds",
            "avg_update_bytes": float(np.mean([r["update_bytes"] for r in part])) if part else None,
            "avg_train_seconds_per_round": float(np.mean([r["train_seconds"] for r in part])) if part else None,
            "at_best_round": {k: at_best.get(k) for k in (
                "local_train_loss", "local_train_accuracy", "local_val_loss", "local_val_accuracy",
                "global_on_local_val_loss", "global_on_local_val_accuracy", "aggregation_weight")},
            "best_round": best_round,
        })
    save_json(client_records, config.OUT["clients"])
    save_json(history, config.OUT["history"])

    federated = {"algorithm": args.algorithm, "mu": mu, "best_round": best_round, "selection_criterion": "highest federated validation accuracy (hospitals score locally)",
                 "rounds": args.rounds, "local_epochs": args.local_epochs, "clients_per_round": stats["clients_per_round"],
                 "training_seconds": stats["training_seconds"], "test": fed_test, "test_accuracy_ci95": [lo, hi],
                 "final_round_test": history[-1]["global_test"]}
    save_json(federated, config.OUT["federated"])

    def row(m, ci):
        return {"accuracy": m["accuracy"], "precision_macro": m["precision_macro"], "recall_macro": m["recall_macro"],
                "f1_macro": m["f1_macro"], "loss": m["loss"], "accuracy_ci95": ci}

    comparison = {"test_images": fed_test["num_samples"], "federated": row(fed_test, [lo, hi]),
                  "federated_training_seconds": stats["training_seconds"]}
    if central:
        comparison["centralized"] = row(central["test"], central["test_accuracy_ci95"])
        comparison["centralized_training_seconds"] = central["training_seconds"]
        comparison["difference_federated_minus_centralized"] = {
            k: comparison["federated"][k] - comparison["centralized"][k]
            for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "loss")}
        comparison["centralized_raw_upload_bytes"] = comm["centralized_alternative_bytes"]
    comparison["federated_training_transfer_bytes"] = comm["total_training_transfer_bytes"]

    generated = datetime.now().isoformat(timespec="seconds")
    experiment = {
        "generated_at": generated,
        "environment": {"python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
                        "platform": platform.platform()},
        "config": {**{k: v for k, v in vars(args).items()}, "grayscale": not args.rgb,
                   "centralized_epochs": (central or {}).get("epochs")},
        "dataset": {k: report[k] for k in ("structure", "valid_images", "corrupted_images", "num_classes", "class_names",
                                            "class_counts", "total_bytes")},
        "partition": {"mode": manifest["partition"], "seed": manifest["seed"], "cleaning": manifest["cleaning"],
                      "test_source": manifest["test"]["source"],
                      "client_train_class_counts": {c["name"]: c["train"]["class_counts"] for c in manifest["clients"]}},
        "federated": federated, "centralized": central, "comparison": comparison, "communication": comm,
    }
    save_json(experiment, config.OUT["experiment"])

    r = comm["ratios"]
    headline = {
        "generated_at": generated, "model_status": "Trained",
        "num_clients": len(clients), "rounds": args.rounds, "best_round": best_round,
        "test_accuracy": fed_test["accuracy"], "test_f1_macro": fed_test["f1_macro"], "test_images": fed_test["num_samples"],
        "num_classes": len(class_names), "class_names": class_names,
        "partition_mode": args.partition, "model_parameters": n_params,
        "avg_update_bytes": comm["avg_update_bytes"], "raw_data_bytes_total": comm["raw_data_bytes_total"],
        "total_training_transfer_bytes": comm["total_training_transfer_bytes"],
        "round_transfer_to_raw_ratio": r["round_transfer_to_total_raw"],
        "total_transfer_to_raw_ratio": r["total_training_transfer_to_total_raw"],
    }
    save_json(headline, config.OUT["metrics"])
    save_json({**model_cfg, "trained_at": generated, "best_round": best_round, "trainable_parameters": n_params,
               "algorithm": args.algorithm, "mu": mu,
               "architecture": architecture_summary(server_model)}, config.MODEL_CONFIG_PATH)

    if args.tag:
        import shutil
        dest = config.ROOT / f"results_{args.tag}"
        # a federated-only run must not carry a centralized model left over from an earlier run
        skip = shutil.ignore_patterns("centralized_*") if args.skip_centralized else None
        if dest.exists():
            shutil.rmtree(dest)
        for sub_dir in ("outputs", "models"):
            shutil.copytree(config.ROOT / sub_dir, dest / sub_dir, ignore=skip)
        print(f"Copied outputs/ and models/ to {dest.name}/")

    print(f"\nDone. Raw data held by hospitals: {format_bytes(comm['raw_data_bytes_total'])}; "
          f"one model update: {format_bytes(comm['avg_update_bytes'])}; "
          f"total federated training traffic: {format_bytes(comm['total_training_transfer_bytes'])}.")
    print("Launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
