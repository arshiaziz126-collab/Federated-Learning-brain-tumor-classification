# NeuroFed AI

**Federated Learning Approach for Brain Tumour Classification with Privacy Preservation**
*Smarter Networks. Healthier Tomorrow.*

A final-year B.Tech CSE project. Three simulated hospitals (Hospital A, B, C) each train a CNN on their own
brain-MRI images. Only model parameters are sent to a server, which combines them with sample-count-weighted
**FedAvg**. A centralized baseline, communication accounting, a privacy audit and a Streamlit dashboard are included.

> Research prototype. Not a clinical diagnostic tool. It requires clinical validation before real-world medical use.

## Team scope

This repository covers: federated learning, privacy preservation, distributed hospital training, FedAvg,
communication efficiency, federated vs centralized comparison, privacy/security analysis and research analytics.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 1. Put the Kaggle brain-tumour MRI dataset in data/brain_tumor_dataset/  (any layout, see below)
python inspect_dataset.py          # optional: see what is in the dataset
python federated_train.py          # trains federated + centralized, writes outputs/ and models/
streamlit run app.py               # opens the dashboard
```

Training is deliberately **separate** from the app. `streamlit run app.py` never trains anything; before training
has been run, the pages show what to do instead of fake numbers.
<img width="1920" height="1080" alt="Screenshot 2026-09-20 151901" src="https://github.com/user-attachments/assets/abefe465-0b51-4ba8-bac0-e5c0590beb46" />


## Dataset

Download the Kaggle brain-tumour MRI dataset yourself and extract it into `data/brain_tumor_dataset/`.
Nothing about its structure is assumed:

- classes are the folder names found on disk (not hardcoded),
- a wrapper folder such as `archive/` is handled automatically,
- if it ships `Training/` and `Testing/` folders, `Testing/` becomes the held-out test set; otherwise a stratified,
  seeded hold-out is created,
- `python inspect_dataset.py` reports classes, image counts, formats, colour modes, dimensions, class balance,
  corrupted files and exact duplicates (including duplicates shared between train and test) and saves
  `outputs/dataset_report.json`.

Before splitting, exact duplicate images are removed (`--no-dedupe` to keep them) so that no image appears in two
hospitals or in both training and test data.

<img width="1920" height="1080" alt="Screenshot 2026-09-20 151930" src="https://github.com/user-attachments/assets/226f2478-e91b-45a5-b746-24dc787b04f4" />

## What `federated_train.py` does

1. **Inspect** the dataset (real statistics only).
2. **Partition** the training images between hospitals with a seeded Dirichlet split (non-IID, `--alpha`, smaller =
   more uneven) or `--partition iid`. Each hospital gets its own physical folder in `data/federated_clients/`,
   with a local train/validation split. The generated class distribution is printed and saved.
3. **Load** each hospital's images from its own folder.
4. **Federated training**, per round: the server serialises the global parameters to bytes -> each hospital trains
   locally and returns serialised parameters + scalar metrics -> the server validates the message shape/dtype ->
   **FedAvg** `w = sum(n_k * w_k) / sum(n_k)` -> hospitals score the new global model on their own validation data
   (only loss/accuracy/count are returned) -> the held-out test set is scored for reporting. The round with the best
   *federated validation* accuracy is kept; the test set is never used to choose it.
5. **Centralized baseline**: the same images pooled, same architecture/optimiser/batch size/learning rate, same test
   set, epochs = rounds x local epochs by default.
6. **Write results**: metrics, communication, privacy report.
<img width="1920" height="1080" alt="Screenshot 2026-09-20 151946" src="https://github.com/user-attachments/assets/51c1ffd8-e263-436a-8760-b3d3e330d588" />

Useful options (`python federated_train.py --help` lists all):

| Option | Default | Meaning |
|---|---|---|
| `--rounds` | 15 | federated rounds |
| `--local-epochs` | 2 | epochs per hospital per round |
| `--num-clients` | 3 | number of hospitals |
| `--partition` / `--alpha` | dirichlet / 0.5 | data split between hospitals |
| `--client-fraction` | 1.0 | share of hospitals selected per round |
| `--img-size` | 128 | images are resized to N x N |
| `--rgb` | off | 3 channels instead of grayscale |
| `--skip-centralized` | off | federated only |
| `--seed` | 42 | partition seed (training itself is not bit-for-bit deterministic) |

Quick trial: `python federated_train.py --rounds 5 --local-epochs 1 --img-size 96`.
Training time depends heavily on your CPU/GPU; with the defaults on a laptop CPU expect on the order of an hour or
more for the federated run plus a similar time for the centralized baseline (a rough estimate extrapolated from a
single-core test, not a measured figure for your dataset).
<img width="1920" height="1080" alt="Screenshot 2026-09-20 152002" src="https://github.com/user-attachments/assets/d2beae0f-b3b6-4e2c-aca4-dcdd24b879b1" />

## Outputs

`models/`: `global_model.keras` (federated), `centralized_model.keras`, `model_config.json` (class names, input size).

`outputs/`: `dataset_report.json`, `partition_manifest.json`, `federated_history.json` (per-round, per-hospital),
`client_metrics.json`, `federated_metrics.json`, `centralized_metrics.json`, `communication_metrics.json`,
`privacy_report.json`, `experiment_results.json`, `metrics.json`.

## Dashboard pages

Home, Diagnose, Federated Network, Federated Analytics, Privacy & Security, Centralized vs Federated, About.
Every value shown is read from the files above. The comparison page presents measured differences and does not
declare either approach better.

## Privacy: what is and is not implemented

Implemented and checked in code:

- **Data locality**: each hospital trains from its own folder only; images are never placed in a message.
- **Parameter-only messages**: hospitals and server exchange serialised `bytes`; the server rejects any message
  whose tensor count, shapes or dtype do not match the model.
- **Isolation audit**: hospital datasets are disjoint by content hash, test images are in no hospital, local
  validation data is not used for local training.

**Not implemented** (and not claimed anywhere): differential privacy, secure aggregation, encryption of updates,
homomorphic encryption. Model updates can leak information about training data. Hospitals are simulated inside one
Python process; there is no real network.

## Communication numbers

Message sizes are the real serialised byte lengths. "Raw data" is the on-disk size of the image files the hospitals
hold. Over many rounds, model traffic can exceed the size of a one-off image upload; the app reports the ratio as
measured and says so when that happens. Federated learning is used here for data locality, not to save bandwidth.

## Early screening wording

The dataset supplies class labels, not tumour stage or grade, so the system is described as an *AI-assisted early
screening research prototype*. It does not predict stage, grade or growth rate.

## Project layout

```
app.py                  Streamlit entry point (navigation + theme)
federated_train.py      training pipeline (federated + centralized + reports)
inspect_dataset.py      dataset inspection utility
config.py               paths, defaults, file names
backend/
  dataset.py            discovery, inspection, cleaning, hospital partitioning
  preprocessing.py      image loading (shared by training and inference)
  model.py              CNN + parameter get/set (swap the model here)
  local_training.py     HospitalClient (local training, local evaluation)
  federated.py          FedAvg + federated round loop
  centralized.py        centralized baseline
  evaluation.py         accuracy / precision / recall / F1 / confusion matrix / Wilson interval
  communication.py      message packing + byte accounting
  privacy.py            payload validation, isolation audit, privacy report
  inference.py          prediction for the Diagnose page
ui/  views/             dashboard theme, components and the seven pages
data/  models/  outputs/
```

To use a different network, add a `build_*` function to `MODEL_REGISTRY` in `backend/model.py` and run with
`--arch <name>`. It must take raw 0-255 pixels and end in a softmax.

## Notes

- Tested with Python 3.12, TensorFlow 2.21 / Keras 3.15, Streamlit 1.64.
- - The interface uses Cormorant Garamond (headings) and Source Sans 3 (text) from Google Fonts; offline it falls back to system fonts.
- Repeat runs with different `--seed` values before drawing conclusions from small differences; one run is one sample.

## Update: FHIR report, Medical Journal Pro theme, FedProx option

| Area | Change | File |
|---|---|---|
| FHIR | Every prediction can be exported as an HL7 FHIR R4 Bundle: pseudonymous Patient, Observation (predicted class + one component per class probability), DiagnosticReport (LOINC 24590-2 "MR Brain", status *preliminary*) with a printable HTML report attached. | `backend/fhir.py`, `backend/report.py` |
| Dashboard | Diagnose page: choose federated or centralized model, prediction with class probabilities, FHIR bundle preview and downloads. Medical Journal Pro theme (Cormorant Garamond headings, Source Sans 3 interface). | `views/diagnose.py`, `ui/theme.py` |
| Training (optional) | `--algorithm fedprox --mu 0.01` adds the FedProx proximal term (Li et al., 2020); `--tag NAME` copies each run to `results_NAME/`. Default training is unchanged FedAvg. | `federated_train.py`, `backend/local_training.py` |

`results_v1/` keeps the results reported in the paper (federated 63.6 %, centralized 85.0 %).

The dataset is the public Kaggle Brain Tumor MRI Dataset (M. Nickparvar); it is not included in this
repository. Hospitals A, B and C are simulated. This is a research prototype and must not be used for
clinical diagnosis.
