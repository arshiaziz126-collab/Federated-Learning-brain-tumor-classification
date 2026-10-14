# NeuroFed AI: training code (Python)

This folder holds the Python and TensorFlow code that trained the models used on the website.

| File | What it does |
|---|---|
| `inspect_dataset.py` | Checks the dataset and finds duplicate images |
| `backend/dataset.py`, `backend/preprocessing.py` | Splits the data across Hospitals A, B and C (Dirichlet, non-IID) and prepares 128 × 128 grayscale images |
| `backend/model.py` | The CNN (299,892 parameters, group normalisation) |
| `backend/local_training.py` | Training inside each hospital |
| `backend/federated.py` | FedAvg and FedProx on the server |
| `backend/centralized.py` | Centralized baseline on pooled data |
| `backend/evaluation.py` | Accuracy, F1, confusion matrix |
| `backend/communication.py` | Communication cost per round |
| `backend/privacy.py` | Isolation and leakage checks |
| `backend/fhir.py`, `backend/report.py` | HL7 FHIR R4 bundle and report |
| `federated_train.py` | Runs the whole experiment |
| `tools/export_web_model.py` | Exports the trained weights for the website |

See `HOW_TO_RUN.txt` for the commands. The dataset is not included.
