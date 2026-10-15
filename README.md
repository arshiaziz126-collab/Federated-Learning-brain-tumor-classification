# NeuroFed AI

**Privacy-preserving federated learning for brain tumour MRI classification, with HL7 FHIR R4 reporting.**

Three simulated hospitals train one shared CNN without ever sharing a patient scan. This repository is the live web demo of that system: the trained model runs directly in your browser, so an uploaded MRI image never leaves your device.

**Live demo:** https://neurofed-ai.vercel.app
<img width="1920" height="899" alt="Screenshot 2026-10-15 142727" src="https://github.com/user-attachments/assets/9bceb708-e6e8-405f-aa0a-a8f942a548be" />

**Training code (Python, TensorFlow):** [`python/`](python) folder

> Research prototype for academic use. Not a clinical diagnostic tool. It requires clinical validation before any real-world medical use.

---

## What it does

- **Diagnose:** upload a brain MRI slice and get a prediction across four classes (glioma, meningioma, pituitary tumour, no tumour) with class probabilities. Switch between the federated model and the centralized baseline to compare them on the same image.
- **HL7 FHIR R4 export:** download each prediction as a FHIR bundle (Patient with pseudonym only, Observation, DiagnosticReport with LOINC 24590-2 "MR Brain"), or as a printable report. Both are marked *preliminary*.
- **Federated network:** how the data is split across Hospitals A, B and C, and what each hospital reported in every training round.
- **Federated analytics:** accuracy and loss per round, per-hospital performance, communication cost, per-class metrics and the confusion matrix.
- **Privacy and security:** what stays inside each hospital, what is shared, the automated isolation checks, and what is not yet implemented.
- **Centralized vs federated:** the same model trained both ways on the same data, plus a FedAvg vs FedProx comparison.
- <img width="1920" height="869" alt="Screenshot 2026-10-15 142841" src="https://github.com/user-attachments/assets/822451f8-cc97-4203-8055-d9ee788489bd" />


Every number on the site is read from the result files of the actual training run. Nothing is hard-coded.

## Results

Held-out test set of 1,584 images, never used for training or model selection.

| Model | Accuracy | Macro F1 | Raw images moved |
|---|---|---|---|
| Centralized baseline (pooled data) | 85.0% | 84.5% | all 5,429 |
| Federated, FedAvg | 63.6% | 55.1% | 0 |
| Federated, FedProx (μ = 0.01) | 62.9% | 54.4% | 0 |
<img width="1920" height="885" alt="Screenshot 2026-10-15 142855" src="https://github.com/user-attachments/assets/3a0c327c-cab9-4c35-a042-ad2d230ca603" />


**Communication:** one model update is about **1.21 MB**, against **126.5 MB** of raw MRI data held by the hospitals.

**Main finding:** the data was split between hospitals in a deliberately uneven (non-IID) way, as happens in real life. Hospital C holds 97.6% of all meningioma training images. When the server averages the three hospitals' models, that knowledge is diluted by the two hospitals that have almost none, so the federated model rarely recognises meningioma. FedProx, which limits how far each hospital drifts, does not close this gap. Recovering a class held by a single site needs methods that handle heterogeneity more directly, such as SCAFFOLD or personalised federated learning.

## How it works

```
Hospital A ──┐
Hospital B ──┼──> local training (2 epochs) ──> model update (1.21 MB) ──> FedAvg on server ──> global model
Hospital C ──┘          images stay on site          parameters only                           (15 rounds)
```

1. The server sends the current global model to each hospital.
2. Each hospital trains it on its own MRI images.
3. Each hospital sends back only its updated parameters, never images.
4. The server averages them, weighted by each hospital's number of images (FedAvg).
5. The round with the best federated validation accuracy becomes the final model.

**Model:** a compact CNN with 299,892 parameters, 128 × 128 grayscale input, four convolution blocks with group normalisation, global average pooling and a softmax over four classes.
<img width="1920" height="909" alt="Screenshot 2026-10-15 142910" src="https://github.com/user-attachments/assets/3ed9e48e-3193-4e71-9741-111dd91d283c" />

**In-browser inference:** the trained weights are exported from Keras and run in plain JavaScript. Image preprocessing reproduces the Python pipeline exactly, so predictions match the Python model (tested: identical inputs, class probabilities within 0.0001).
<img width="1920" height="883" alt="Screenshot 2026-10-15 142933" src="https://github.com/user-attachments/assets/b8b2c6f8-1f2d-4c33-b25f-860360e49f95" />


## Dataset

Public [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by M. Nickparvar on Kaggle: 7,200 images, four classes. After removing 187 exact duplicates, 7,013 images were used (5,429 for the hospitals, 1,584 for testing). The dataset is not included in this repository. Hospitals A, B and C are simulated.

## Privacy

Implemented:
- Raw MRI images stay in each hospital's own folder; only model parameters and a few scalar metrics are shared.
- The server validates every update (tensor count, shapes and data type) before averaging.
- Automated checks confirm that hospital datasets do not overlap, that no test image is in any hospital, and that validation images are never used for training.
- FHIR reports identify the patient by pseudonym only.
- In this web demo, uploaded images are processed in the browser and never sent to a server.

Not yet implemented: encryption in transit (TLS), secure aggregation and differential privacy. These are listed as future work.
<img width="1920" height="888" alt="Screenshot 2026-10-15 142945" src="https://github.com/user-attachments/assets/20fa3c21-e973-4a6d-b42f-876ff4b728fd" />


## Repository layout

```
index.html        app shell
css/style.css     white-on-black theme (Instrument Serif and DM Sans)
js/app.js         pages, routing and the Diagnose flow
js/model.js       the CNN and Pillow-compatible preprocessing in JavaScript
js/fhir.js        HL7 FHIR R4 bundle and printable report
js/charts.js      SVG charts
models/           exported weights of the federated and centralized models
data/             result files from the training run
vercel.json       Vercel settings
python/           Python training code: dataset split, CNN, FedAvg/FedProx,
                  centralized baseline, evaluation, privacy checks, FHIR
```

## Run locally

```bash
python -m http.server 8000
```

Open http://localhost:8000. Do not open `index.html` by double-clicking it: browsers block loading the model files from a local file.

## Deploy

Import this repository in [Vercel](https://vercel.com), choose the framework preset **Other**, leave the build command empty and deploy. There is no build step.

## Tech stack

**Training:** Python, TensorFlow and Keras, NumPy, scikit-learn, Pillow (see the `python/` folder)
**Web demo:** HTML, CSS and JavaScript with no framework, hosted on Vercel
**Interoperability:** HL7 FHIR R4, LOINC

## Team

Arshi, Shagun Kumari, Vanshika Chauhan, Vimmi Tyagi
Guide: Rajesh Thakur
Department of Computer Science and Engineering, Meerut Institute of Engineering and Technology (MIET), Meerut

Final-year B.Tech CSE major project. The accompanying research paper describes the method and results in full.
