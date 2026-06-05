# ⚡ EV Battery Life Prediction Using ANN (Keras)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Sequential_API-red?style=flat-square&logo=keras)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Dataset: NASA](https://img.shields.io/badge/Dataset-NASA_PCoE_Battery-blueviolet?style=flat-square)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Author:** [wittyswayam](https://github.com/wittyswayam)  
> **Repository:** `EV_Battery_Life_Prediction_Using_ANN_Keras`

---

## 📋 Project Overview

**EV Battery Life Prediction Using ANN (Keras)** is an end-to-end machine learning system that uses a feed-forward Artificial Neural Network to predict the **ambient operating temperature** of lithium-ion EV battery cells from electrochemical cycle measurements. The training data originates from the **NASA Prognostics Center of Excellence (PCoE) Battery Dataset** — a gold-standard experimental corpus covering 7,565 charge, discharge, and impedance spectroscopy cycles across 34 distinct Li-ion cells (B0025–B0056) tested at 4 °C, 24 °C, and 44 °C.

### Real-World Problem Solved

Electric vehicle fleets accumulate hundreds of thousands of battery cycles across a wide range of environmental temperatures. Determining the ambient temperature at which a specific cycle occurred — from electrochemical measurements alone — is a core step in battery degradation modelling. Traditional approaches require sensor-dense physical test rigs or manual log cross-referencing. This system learns the electrochemical signature of each temperature regime directly from cycle data, enabling automated, software-only temperature inference that integrates into broader state-of-health pipelines.

### Target Users

- **ML Engineers** building battery degradation baselines or benchmarking ANN regression approaches on electrochemical datasets.
- **EV Fleet Operators** seeking lightweight, offline-capable inference tools for battery condition estimation without cloud dependencies.
- **Battery R&D Teams** at automotive OEMs or cell manufacturers needing a reproducible, extensible starting point grounded in real NASA experimental data.
- **Data Scientists & Students** studying applied deep learning on tabular sensor data.

### Architecture & Design Rationale

The system is intentionally structured as two clearly separated phases — an **offline training pipeline** (Jupyter Notebook) and an **online inference application** (Streamlit) — connected solely through serialised artefacts on disk. This separation ensures:

- **Training-serving consistency:** Every preprocessing transformation (label encoding, feature scaling) is fit exclusively on training data and persisted, guaranteeing byte-identical preprocessing at inference time and eliminating training-serving skew.
- **Zero-infrastructure deployment:** The full inference stack runs in a single Python process with no external databases, message queues, or cloud APIs.
- **Extensibility:** Swapping the ANN for an LSTM, XGBoost, or transformer architecture requires only changes to the training notebook; `app.py` continues to load whichever artefacts are serialised.

---

## 🔬 Core Features

### 1. Fail-Safe Data Ingestion
`metadata.csv` (7,565 rows × 10 columns) is ingested with `pd.read_csv()`. Columns `Capacity`, `Re`, and `Rct` contain non-numeric sentinel strings from dataset aggregation and are coerced with `pd.to_numeric(errors='coerce')`, converting unparseable values to `NaN` rather than raising exceptions. This implements the Tolerant Reader pattern — all rows are preserved at ingestion; the imputation decision is deferred to an explicit, auditable step.

### 2. Mean Imputation Across Heterogeneous Cycle Types
Three columns carry structural missingness because each column is only populated for one cycle type:

| Column | Populated For | Present Rows | Missing Rate |
|---|---|---|---|
| `Capacity` | `discharge` only | 2,794 | ~63% |
| `Re` | `impedance` only | 1,956 | ~74% |
| `Rct` | `impedance` only | 1,956 | ~74% |

Missing values are filled with `column.mean()` — a single-pass, O(n) operation compatible with the downstream `MinMaxScaler`.

### 3. Feature Engineering & Dimensionality Reduction
Five identifier and metadata columns (`start_time`, `battery_id`, `test_id`, `uid`, `filename`) are dropped before modelling. These columns encode data provenance, not battery physics — retaining them would cause the model to memorise cell identities rather than learn electrochemical relationships. The resulting feature matrix `X` contains four columns: `type` (encoded), `Capacity`, `Re`, `Rct`.

### 4. Categorical Encoding with Artefact Persistence
The `type` column (`charge`, `discharge`, `impedance`) is ordinally encoded using `sklearn.LabelEncoder.fit_transform()`. The fitted encoder object is serialised to `label_encoder.pkl` for deterministic replay at inference time, mapping: `charge→0`, `discharge→1`, `impedance→2`.

### 5. Leakage-Free Feature Scaling
`MinMaxScaler` is fit **exclusively on `X_train`** and then applied via `.transform()` to both `X_train` and `X_test`. The fitted scaler is serialised to `scaler.pkl`. At inference time, `app.py` loads this exact fitted scaler — it never re-fits on inference inputs, preventing data leakage.

### 6. Keras Sequential ANN (Regression)
A compact multi-layer perceptron is built with the `tf.keras.Sequential` API:

| Layer | Type | Units | Activation | Trainable Params |
|---|---|---|---|---|
| Input | Dense | 64 | ReLU | 320 |
| — | Dropout | — | — (rate=0.2) | 0 |
| Hidden | Dense | 32 | ReLU | 2,080 |
| — | Dropout | — | — (rate=0.2) | 0 |
| Output | Dense | 1 | Linear | 33 |
| **Total** | | | | **2,433** |

The pyramidal topology (64→32→1) progressively abstracts electrochemical features into a scalar temperature prediction. ReLU activations handle non-linearities; Dropout(0.2) after each hidden layer regularises the model on the ~6,000-row training set. The linear output activation is mandatory for unbounded regression.

### 7. Artefact Serialisation Pipeline
After training, all inference dependencies are written to disk:
- `battery_life_model.h5` — Keras HDF5 native save (consumed by `app.py`)
- `battery_life_model.pkl` — Pickle serialisation of the Keras model (alternative format)
- `scaler.pkl` — Fitted `MinMaxScaler`
- `label_encoder.pkl` — Fitted `LabelEncoder`

### 8. Streamlit Prediction Interface
`app.py` provides a reactive browser-based inference UI. It loads all artefacts once at session start, renders a four-field input form, and calls `predict_battery_life()` on button click — delivering a complete, no-API-required prediction tool in ~45 lines of Python.

---

## 🏗️ System Architecture

The system operates in two distinct phases connected through persisted artefacts:

```
╔══════════════════════════════════════════════════════════════╗
║              OFFLINE TRAINING PIPELINE (Notebook)            ║
╚══════════════════════════════════════════════════════════════╝

  metadata.csv (7,565 × 10)
       │
       ▼
  [Ingestion]  pd.read_csv() + pd.to_numeric(errors='coerce')
       │
       ▼
  [Pruning]    Drop: start_time, battery_id, test_id, uid, filename
       │
       ▼
  [Imputation] fillna(column.mean()) on Capacity, Re, Rct
       │
       ▼
  [Encoding]   LabelEncoder.fit_transform(df['type'])
       │
       ▼
  [Split]      X_train (6,052 rows) / X_test (1,513 rows) | random_state=42
       │
       ▼
  [Scaling]    MinMaxScaler.fit_transform(X_train) / .transform(X_test)
       │
       ▼
  [Training]   Sequential ANN  4→64(ReLU)→Drop→32(ReLU)→Drop→1(linear)
               Optimizer: Adam | Loss: MSE | Epochs: 150 | Batch: 32
               validation_data=(X_test_scaled, y_test)
       │
       ▼
  [Serialise]  battery_life_model.h5 + .pkl | scaler.pkl | label_encoder.pkl


╔══════════════════════════════════════════════════════════════╗
║            ONLINE INFERENCE PIPELINE (Streamlit)             ║
╚══════════════════════════════════════════════════════════════╝

  App Start → load_model(.h5) + pickle.load(scaler, encoder)
       │
       ▼
  Browser UI   Discharge Type (selectbox) | Capacity | Re | Rct (number_input)
       │  [button click]
       ▼
  label_encoder.transform([type_discharge])   ← transform only, never re-fit
       │
       ▼
  np.array([[type_encoded, Capacity, Re, Rct]])  shape: (1, 4)
       │
       ▼
  scaler.transform(X_input)                  ← applies training-fitted bounds
       │
       ▼
  model.predict(X_scaled)                    ← forward pass: 4→64→32→1
       │
       ▼
  st.write(f"Predicted battery life: {result[0]} units")
```

---

## 🛠️ Tech Stack

| Category | Technology | Role |
|---|---|---|
| **Language** | Python ≥3.9 | Primary development and runtime language |
| **Deep Learning** | TensorFlow / Keras 2.x | Sequential ANN definition, training, and `.h5` model persistence |
| **ML Preprocessing** | scikit-learn ≥1.0 | `MinMaxScaler` (feature normalisation) and `LabelEncoder` (categorical encoding) |
| **Data Manipulation** | Pandas ≥1.3 | DataFrame ingestion, type coercion, NaN imputation, column dropping |
| **Numerical Computing** | NumPy ≥1.21 | Feature vector assembly for single-sample inference |
| **Visualisation** | Matplotlib ≥3.4 | Training vs. validation loss curve plotting in the notebook |
| **Web Application** | Streamlit ≥1.0 | Single-file reactive browser UI for prediction without a separate frontend or API layer |
| **Serialisation** | pickle (stdlib) | Persistence of sklearn preprocessing objects (scaler, encoder) |
| **Model Persistence** | Keras HDF5 (`.h5`) | TF-native model serialisation preserving architecture, weights, and optimiser state |
| **Dataset** | NASA PCoE Battery (`metadata.csv`) | 7,565-row electrochemical measurement corpus for 34 Li-ion cells |
| **Notebook Runtime** | Jupyter Notebook ≥6.0 | Interactive training, EDA, and visualisation environment |

---

## 📁 Repository Structure

```
EV_Battery_Life_Prediction_Using_ANN_Keras-main/
│
├── EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb
│   └── Master training notebook. Contains all preprocessing steps, ANN
│       definition and training (150 epochs, batch 32), Matplotlib loss plots,
│       model evaluation, and all four artefact serialisation calls.
│
├── app.py
│   └── Streamlit prediction application (~45 lines). Loads battery_life_model.h5,
│       scaler.pkl, and label_encoder.pkl at startup. Renders a 4-field form;
│       calls predict_battery_life() on button press; displays result.
│
├── metadata.csv
│   └── Primary dataset. 7,565 rows × 10 columns. NASA PCoE Battery cycling
│       data for batteries B0025–B0056. Columns: type, start_time,
│       ambient_temperature, battery_id, test_id, uid, filename,
│       Capacity, Re, Rct.
│
├── battery_life_model.pkl
│   └── Pickle-serialised Keras model (~59 KB). Alternative format; app.py
│       uses battery_life_model.h5 instead.
│
├── scaler.pkl
│   └── Pickle-serialised MinMaxScaler (~757 bytes) fitted on X_train.
│       Must be loaded and applied (transform only) at inference time.
│
├── label_encoder.pkl
│   └── Pickle-serialised LabelEncoder (~275 bytes). Maps charge/discharge/
│       impedance → 0/1/2. Must be loaded for inference.
│
├── battery_life_model.h5          ← Generated by notebook; NOT in ZIP
│   └── Keras HDF5 model file consumed by app.py via load_model().
│       Regenerate by running the training notebook.
│
└── README.md
```

> **⚠️ Important:** `battery_life_model.h5` is a binary training output and is **not** included in the repository ZIP. You must run the notebook to generate it before launching `app.py`.

---

## ⚙️ Installation & Environment Setup

### Prerequisites

| Requirement | Min Version |
|---|---|
| Python | 3.9 |
| pip | 21.0 |
| Git | 2.30 |

### Step-by-Step Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/wittyswayam/EV_Battery_Life_Prediction_Using_ANN_Keras.git
cd EV_Battery_Life_Prediction_Using_ANN_Keras
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows CMD
.venv\Scripts\activate.bat
```

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install tensorflow>=2.10 \
            scikit-learn>=1.0 \
            pandas>=1.3 \
            numpy>=1.21 \
            matplotlib>=3.4 \
            streamlit>=1.10 \
            jupyter>=1.0
```

**4. Run the training notebook to generate `battery_life_model.h5`**
```bash
jupyter notebook
# Open: EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb
# Run: Kernel → Restart & Run All
# Training takes ~2–8 minutes on CPU.
```

**5. Verify all artefacts are present**
```bash
ls -lh battery_life_model.h5 battery_life_model.pkl scaler.pkl label_encoder.pkl
# All four files must exist before launching the app.
```

**6. Launch the Streamlit app**
```bash
streamlit run app.py
# Open http://localhost:8501 in your browser.
```

### Docker Setup

Before building, ensure `battery_life_model.h5` has been generated by the notebook.

```dockerfile
# Dockerfile
FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py battery_life_model.h5 scaler.pkl label_encoder.pkl ./
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t ev-battery-prediction:latest .
docker run -d --name ev-battery-app -p 8501:8501 ev-battery-prediction:latest
# Access at http://localhost:8501
```

### Common Setup Issues

| Problem | Cause | Fix |
|---|---|---|
| `FileNotFoundError: battery_life_model.h5` | `.h5` not generated yet | Run the notebook first (Kernel → Restart & Run All) |
| `ValueError` on `label_encoder.transform` | Unknown cycle type string passed | Only `'charge'`, `'discharge'`, `'impedance'` are valid |
| `sklearn` version mismatch on `.pkl` load | `scaler.pkl` created with a different sklearn version | Retrain with your installed sklearn version |
| Streamlit widgets reset on button click | Expected Streamlit behaviour | Inputs persist; re-enter values if needed |

---

## 🚀 Usage Guide

### Train the Model (Notebook)

```bash
jupyter notebook
# Open EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb
# Execute all cells sequentially.
# Output: battery_life_model.h5, battery_life_model.pkl, scaler.pkl, label_encoder.pkl
```

### Run Inference (Streamlit UI)

```bash
streamlit run app.py
```

In the browser:
1. **Discharge Type** → select `discharge`
2. **Capacity** → enter `1.6743` (Ah)
3. **Re** → enter `0.056058` (Ω)
4. **Rct** → enter `0.200970` (Ω)
5. Click **Predict Battery Life**

Expected output: a value near 4.0 or 24.0 (the discrete ambient temperatures in the training dataset).

### Run Inference Programmatically (Python)

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load artefacts
model         = load_model("battery_life_model.h5")
scaler        = pickle.load(open("scaler.pkl",        "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

def predict_battery_life(type_discharge, Capacity, Re, Rct,
                          label_encoder, scaler, model):
    encoded = label_encoder.transform([type_discharge])[0]
    X = np.array([[encoded, Capacity, Re, Rct]])
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)[0]

result = predict_battery_life(
    type_discharge = "discharge",
    Capacity       = 1.6743,
    Re             = 0.056058,
    Rct            = 0.200970,
    label_encoder  = label_encoder,
    scaler         = scaler,
    model          = model
)
print(f"Predicted ambient temperature: {result[0]:.2f} °C")
```

---

## 📖 API & Module Documentation

### `predict_battery_life()` — Core Inference Function

Defined in `app.py` and the training notebook. Encapsulates the full preprocessing + inference chain.

**Signature:**
```python
def predict_battery_life(
    type_discharge: str,         # 'charge' | 'discharge' | 'impedance'
    Capacity:       float,       # Battery capacity in Ah (≥ 0.0)
    Re:             float,       # Electrolyte resistance in Ohms
    Rct:            float,       # Charge-transfer resistance in Ohms
    label_encoder:  LabelEncoder,
    scaler:         MinMaxScaler,
    model:          Sequential
) -> np.ndarray                  # shape (1,) — predicted ambient_temperature °C
```

**Input validation:**

| Parameter | Valid Values | Exception if Invalid |
|---|---|---|
| `type_discharge` | `'charge'`, `'discharge'`, `'impedance'` | `ValueError` from `LabelEncoder` |
| `Capacity` | Finite float ≥ 0.0 | `ValueError` in `scaler.transform()` if NaN |
| `Re`, `Rct` | Any finite float | `ValueError` if NaN or ±infinity |

### Streamlit Application UI Contract (`app.py`)

| Widget | `st.*` Type | Maps To | Range |
|---|---|---|---|
| Discharge Type | `st.selectbox` | `type_discharge: str` | `charge` \| `discharge` \| `impedance` |
| Capacity | `st.number_input` | `Capacity: float` | `[0.0, ∞)` |
| Re | `st.number_input` | `Re: float` | `[−1×10¹², 1×10¹²]` |
| Rct | `st.number_input` | `Rct: float` | `[−1×10¹², 1×10¹²]` |

**Trigger:** `st.button("Predict Battery Life")` — on click, executes the full inference pipeline and renders the result via `st.write()`.

---

## 🧪 Development Workflow

### Linting & Formatting

```bash
pip install flake8 black isort

# PEP 8 compliance
flake8 app.py --max-line-length=120

# Auto-format
black app.py

# Import sorting
isort app.py
```

### Recommended Test Structure

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing
```

```python
# tests/test_inference.py
import pickle, numpy as np, pytest
from tensorflow.keras.models import load_model
from app import predict_battery_life

@pytest.fixture(scope="module")
def artefacts():
    model = load_model("battery_life_model.h5")
    scaler = pickle.load(open("scaler.pkl", "rb"))
    le     = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, le

def test_encoder_classes(artefacts):
    _, _, le = artefacts
    assert set(le.classes_) == {'charge', 'discharge', 'impedance'}

def test_scaler_feature_count(artefacts):
    _, scaler, _ = artefacts
    assert scaler.n_features_in_ == 4

def test_output_shape(artefacts):
    model, scaler, le = artefacts
    result = predict_battery_life('discharge', 1.674, 0.056, 0.201, le, scaler, model)
    assert result.shape == (1,)

def test_prediction_in_physical_range(artefacts):
    model, scaler, le = artefacts
    result = predict_battery_life('discharge', 1.674, 0.056, 0.201, le, scaler, model)
    assert -20.0 <= float(result[0]) <= 80.0

def test_unknown_type_raises(artefacts):
    model, scaler, le = artefacts
    with pytest.raises(Exception):
        predict_battery_life('fast_charge', 1.674, 0.056, 0.201, le, scaler, model)
```

### Recommended GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:  { branches: [main, develop] }
  pull_request: { branches: [main] }
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install tensorflow scikit-learn pandas numpy matplotlib streamlit pytest flake8 black nbconvert
      - run: flake8 app.py --max-line-length=120
      - run: black app.py --check
      - run: jupyter nbconvert --to notebook --execute EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb --ExecutePreprocessor.timeout=600
      - run: pytest tests/ -v
```

---

## 🤝 Contributing Guidelines

### Branch Naming

```
feat/<short-description>      # New feature
fix/<short-description>       # Bug fix
docs/<short-description>      # Documentation only
refactor/<short-description>  # Code restructuring
test/<short-description>      # Test additions
```

### Commit Convention (Conventional Commits)

```
<type>(<scope>): <short description>
```

| Type | When to Use |
|---|---|
| `feat` | New capability (e.g., `feat(app): add batch CSV prediction mode`) |
| `fix` | Bug fix (e.g., `fix(app): correct feature column order in inference`) |
| `docs` | Documentation (e.g., `docs: add Docker deployment section to README`) |
| `perf` | Performance (e.g., `perf(app): add st.cache_resource to artefact loading`) |
| `refactor` | No behaviour change (e.g., `refactor: extract preprocessing into pipeline.py`) |
| `test` | Test additions (e.g., `test: add unit tests for predict_battery_life`) |
| `chore` | Maintenance (e.g., `chore: pin tensorflow to 2.12 in requirements.txt`) |

### PR Process

1. Fork the repository and create a branch from `main`.
2. Ensure the notebook executes end-to-end (`Kernel → Restart & Run All`) and `streamlit run app.py` launches without errors.
3. Pass all linting checks (`flake8`, `black --check`).
4. Open a Pull Request with a description of the change, motivation, and test evidence.
5. A maintainer will review within 5 business days.

---

## ⚡ Performance & Scalability

| Area | Current Implementation | Recommended Extension |
|---|---|---|
| **Artefact Loading** | Loaded at module-level on every Streamlit script re-run | Add `@st.cache_resource` to load artefacts once per server session |
| **Model Size** | 2,433 trainable parameters — <1 MB total | No optimisation required; TF Lite conversion available for mobile/edge deployment |
| **Inference Latency** | Single-sample, synchronous, ~10–50 ms on CPU | For batch use cases, accept CSV upload and vectorise inference with `model.predict(X_batch)` |
| **Training Scale** | CPU-only; ~2–8 minutes for 150 epochs on 6,052 samples | GPU training drop-in via CUDA-enabled TF build; no code changes required |
| **Dataset Scale** | 7,565 rows in a single CSV | For larger datasets, replace `pd.read_csv()` with chunked reading or Parquet ingestion |

**Recommended `@st.cache_resource` pattern:**
```python
@st.cache_resource
def load_artefacts():
    model         = load_model("battery_life_model.h5")
    scaler        = pickle.load(open("scaler.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artefacts()
```
This reduces artefact load time from ~1–2 seconds per interaction to milliseconds after the initial session start.

---

## 🔒 Security & Reliability

| Concern | Current State | Recommended Action |
|---|---|---|
| **Pickle deserialisation** | `scaler.pkl` and `label_encoder.pkl` loaded from local disk | Never load `.pkl` files from untrusted or user-supplied paths; validate SHA-256 checksums in production |
| **Input validation** | Streamlit widget bounds (client-side only) | Add server-side finite-value checks and out-of-distribution warnings before calling `model.predict()` |
| **Model file integrity** | No checksum verification of `.h5` | Store and verify SHA-256 of all artefacts at load time in production deployments |
| **CORS / network exposure** | Streamlit default (local only) | In production, set `STREAMLIT_SERVER_ENABLECORS=false` and proxy behind nginx with TLS |
| **Dependency vulnerabilities** | Not audited | Run `pip audit` regularly; pin all versions in `requirements.txt` |

---

## 🗺️ Roadmap

| Priority | Item | Description |
|---|---|---|
| 🔴 High | `@st.cache_resource` for artefact loading | Prevents reloading model on every widget interaction |
| 🔴 High | Formal `pytest` test suite | Unit tests for preprocessing pipeline and inference function |
| 🔴 High | Temporal train/test split | Split by time rather than randomly to prevent future-data leakage across battery cycles |
| 🟡 Medium | LSTM / GRU model | Leverage the sequential structure of discharge cycles for improved temporal accuracy |
| 🟡 Medium | State of Health (SOH) target | Derive SOH = Capacity / NominalCapacity per `battery_id` and predict it directly |
| 🟡 Medium | REST API layer (FastAPI) | Expose `/api/v1/predict` endpoint for programmatic fleet management integration |
| 🟢 Low | MLflow experiment tracking | Log hyperparameters, metrics, and artefacts per training run |
| 🟢 Low | Kubernetes deployment manifests | Helm chart for horizontally scalable production deployment |

### Known Limitations

- **Random split:** The 80/20 split is random, not temporal. This risks future-data leakage across sequential battery cycles from the same cell.
- **Mean imputation across cycle types:** `Capacity`, `Re`, and `Rct` are only physically meaningful for their respective cycle types; cross-cycle mean imputation introduces synthetic cross-type signal.
- **Discrete target variable:** `ambient_temperature` takes only three values (4, 24, 44 °C). This is a regression problem with a low-cardinality target — a classification formulation may yield higher accuracy.
- **No input validation at inference:** The app does not warn users if input values fall outside the training distribution.

---

## 📊 Dataset Attribution

This project uses the **NASA Prognostics Center of Excellence (PCoE) Battery Dataset**:

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.  
> URL: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

---

## 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2024 wittyswayam

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

<p align="center">
  Built with ⚡ by <a href="https://github.com/wittyswayam">wittyswayam</a> · Powered by NASA PCoE Battery Data · TensorFlow / Keras · Streamlit
</p>
