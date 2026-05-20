# ⚡ EV Battery Life Prediction Using ANN (Keras)

[![Build Status](https://img.shields.io/github/actions/workflow/status/yourname/EV_Battery_Life_Prediction_Using_ANN_Keras/ci.yml?branch=main&style=flat-square)](https://github.com/yourname/EV_Battery_Life_Prediction_Using_ANN_Keras/actions)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Dataset: NASA](https://img.shields.io/badge/Dataset-NASA%20Battery-blueviolet?style=flat-square)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

---

## 📋 Project Overview

**EV Battery Life Prediction Using ANN (Keras)** is an end-to-end machine learning system that applies an Artificial Neural Network to predict the **ambient operating temperature** of lithium-ion electric vehicle batteries using electrochemical discharge measurement data derived from the NASA Battery Dataset. The system addresses a core pain point in EV fleet management: accurately forecasting battery degradation behaviour without costly, disruptive physical testing — enabling data-driven decisions around preventive maintenance, warranty risk assessment, and charging optimisation.

The project delivers a full, production-oriented pipeline spanning raw data ingestion, automated preprocessing, model training, serialised artefact persistence, and a live web-based inference application built with Streamlit. It targets **data scientists**, **ML engineers**, **EV fleet operators**, and **battery R&D teams** seeking a reproducible, extensible baseline for electrochemical health monitoring.

### 🎯 Key Value Propositions

- **Full Pipeline Reproducibility:** Every transformation applied during training (label encoding, min-max scaling) is serialised alongside the model, guaranteeing byte-identical preprocessing at inference time and eliminating training-serving skew.
- **Robust Electrochemical Feature Set:** Predictions are grounded in three physically meaningful electrochemical signals — Capacity, electrolyte resistance (Re), and charge-transfer resistance (Rct) — rather than superficial time-series proxies.
- **Lightweight Deployment:** The complete inference stack (model + scaler + encoder) runs entirely in-process within a Streamlit application, requiring no external databases, message queues, or cloud infrastructure.
- **Extensible Architecture:** Clear separation between the training notebook, serialised artefacts, and the serving app makes it straightforward to swap in alternative architectures (LSTM, XGBoost) or extend the feature space without rewriting the inference layer.
- **Real Dataset Provenance:** Training data originates from the NASA Prognostics Center of Excellence Battery Dataset, a gold-standard experimental corpus covering charge/discharge/impedance cycles across 34 lithium-ion cells under controlled laboratory conditions.

---

## 🔬 Core Features & Functional Capabilities

### 2.1 Data Ingestion & Raw Dataset Management

**What it does:** The system reads `metadata.csv`, a 7,565-row × 10-column CSV file derived from NASA battery cycling experiments, into a Pandas DataFrame. Each row represents a single battery test event (charge cycle, discharge cycle, or impedance spectroscopy measurement) for one of 34 distinct battery cells (labelled B0025–B0056).

**Technical mechanism:** Standard `pandas.read_csv()` ingestion with subsequent dtype inspection. Because several numeric columns (`Capacity`, `Re`, `Rct`) contain non-numeric sentinel strings introduced during dataset aggregation, the pipeline relies on `pd.to_numeric(..., errors='coerce')` to coerce invalid entries to `NaN` rather than raising parse exceptions. This implements a fail-safe ingestion pattern consistent with the Tolerant Reader design pattern.

**Why this approach:** Forcing parse errors silently to `NaN` preserves the full row count during ingestion and defers the imputation decision to an explicit, auditable preprocessing step rather than silently discarding rows at read time.

---

### 2.2 Automated Missing-Value Imputation

**What it does:** After type coercion, three columns carry substantial `NaN` populations:

| Column | Non-null Count | Missing Rate |
|---|---|---|
| `Capacity` | 2,794 / 7,565 | ~63% |
| `Re` | 1,956 / 7,565 | ~74% |
| `Rct` | 1,956 / 7,565 | ~74% |

Missing values in all three columns are filled with their respective column means via `DataFrame.fillna(column.mean(), inplace=True)`.

**Technical mechanism:** Mean imputation is a single-pass, O(n) operation. Since Re and Rct are only populated for impedance-type rows and Capacity is only populated for discharge-type rows, mean imputation effectively synthesises cross-cycle feature values — a deliberate simplification that maintains a uniform feature matrix shape across all measurement types.

**Why this approach:** Mean imputation is chosen here as the simplest stationary imputer compatible with the downstream `MinMaxScaler`, which requires no `NaN` values. More sophisticated strategies (e.g., KNN imputation grouped by battery_id) are left as future improvements.

---

### 2.3 Dimensionality Reduction via Feature Selection

**What it does:** Five metadata columns that carry no predictive electrochemical signal are dropped before modelling:

```python
df = df.drop(columns=['start_time', 'battery_id', 'test_id', 'uid', 'filename'])
```

**Technical mechanism:** Explicit column exclusion reduces the input dimensionality from 10 to 5 columns (`type`, `ambient_temperature`, `Capacity`, `Re`, `Rct`), after which `ambient_temperature` is isolated as the regression target (y) and the remaining four columns form the feature matrix (X).

**Why this approach:** Identifier columns (`battery_id`, `test_id`, `uid`, `filename`) are not generalisable features — they encode data provenance rather than battery physics. Including them would cause the model to memorise cell identities rather than learn electrochemical relationships. `start_time` is stored as a Python-list-formatted string (e.g., `[2010. 7. 21. 15. 0. 35.093]`) and would require non-trivial parsing; excluding it is the practical choice at this prototype stage.

---

### 2.4 Categorical Encoding (LabelEncoder)

**What it does:** The `type` column contains three string values — `charge`, `discharge`, and `impedance` — representing the nature of the battery cycle. These are ordinally encoded to integers using `sklearn.preprocessing.LabelEncoder`.

**Technical mechanism:**

```python
label_encoder = LabelEncoder()
df['type'] = label_encoder.fit_transform(df['type'])
```

The fitted encoder maps: `charge → 0`, `discharge → 1`, `impedance → 2` (alphabetical default ordering). The fitted encoder object is serialised to `label_encoder.pkl` for deterministic replay at inference time, ensuring the app never re-fits on a different data distribution.

**Why this approach:** LabelEncoder is appropriate here because the downstream scaler (`MinMaxScaler`) treats all features numerically — the ordinal encoding is bounded and immediately rescaled into [0, 1], mitigating any inadvertent ordinal relationship imposed by the integer assignment.

---

### 2.5 Feature Scaling (MinMaxScaler)

**What it does:** All four input features are normalised into the [0, 1] range using `sklearn.preprocessing.MinMaxScaler`. The scaler is fitted exclusively on `X_train` and then applied to both `X_train` and `X_test` without re-fitting — the standard train-only-fit discipline that prevents data leakage.

**Technical mechanism:**

```python
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
```

The fitted scaler is serialised to `scaler.pkl`. At inference time, the Streamlit app loads this exact scaler object and calls `.transform()` on the user-supplied four-feature vector before passing it to the model.

**Why MinMaxScaler over StandardScaler:** MinMaxScaler preserves zero values (important for electrochemical features which can legitimately be zero) and ensures all features are bounded, which is well-suited to the ReLU-activated hidden layers in the ANN where unbounded inputs can slow convergence. The README and `app.py` reference `StandardScaler` in some comments — the ground truth is the serialised `scaler.pkl`, which was created with `MinMaxScaler` in the training notebook.

---

### 2.6 Train–Test Split

**What it does:** The scaled dataset is partitioned into an 80% training set and a 20% held-out test set using a fixed random seed.

**Technical mechanism:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

With 7,565 total samples: `X_train` = 6,052 rows, `X_test` = 1,513 rows.

**Why this approach:** `random_state=42` guarantees reproducibility across runs. The 80/20 split is conventional for tabular regression tasks of this scale. Note that this is a random (non-temporal) split; because the dataset contains sequential battery cycles, a strictly temporal split (earlier cycles train, later cycles test) would be more rigorous and is identified as a future improvement.

---

### 2.7 Artificial Neural Network Architecture (Keras Sequential API)

**What it does:** A feed-forward multi-layer perceptron (MLP) is constructed using the `tensorflow.keras.Sequential` API for regression on the target variable `ambient_temperature`.

**Technical mechanism:** The network architecture is:

| Layer | Type | Units | Activation | Parameter Count |
|---|---|---|---|---|
| Input | Dense | 64 | ReLU | (4 inputs × 64) + 64 = 320 |
| Regularisation | Dropout | — | — | 0 (rate = 0.20) |
| Hidden | Dense | 32 | ReLU | (64 × 32) + 32 = 2,080 |
| Regularisation | Dropout | — | — | 0 (rate = 0.20) |
| Output | Dense | 1 | Linear | (32 × 1) + 1 = 33 |
| **Total** | | | | **2,433 trainable parameters** |

**Why this architecture:** The two-hidden-layer design with decreasing width (64 → 32) is a standard pyramidal topology that progressively abstracts input features into a compact latent representation before the scalar regression head. Dropout at rate 0.2 after each hidden layer provides regularisation, reducing overfitting on the relatively small training set. The linear output activation is mandatory for unbounded regression targets (ambient temperature in °C).

---

### 2.8 Model Compilation & Optimisation

**What it does:** The network is compiled with the Adam optimiser and Mean Squared Error (MSE) as the loss function.

**Technical mechanism:**

```python
model.compile(optimizer='adam', loss='mean_squared_error')
```

Adam (`lr=0.001` default) provides adaptive per-parameter learning rates via first- and second-moment gradient estimates (RMSProp + momentum), making it robust to the heterogeneous feature scales even after MinMaxScaling.

**Why MSE:** MSE is the canonical loss function for continuous regression. It penalises large prediction errors quadratically, which is appropriate here since gross temperature misestimates (e.g., predicting 4°C when the true value is 44°C) are more consequential than small errors.

---

### 2.9 Model Training with Validation Monitoring

**What it does:** The model is trained for 150 epochs with a batch size of 32, using the test split as the validation dataset to monitor generalisation loss throughout training.

**Technical mechanism:**

```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_data=(X_test_scaled, y_test)
)
```

The `history` object records per-epoch `loss` and `val_loss`, which are subsequently plotted with Matplotlib to visually diagnose overfitting or underfitting.

**Why 150 epochs / batch size 32:** With 6,052 training samples and batch size 32, each epoch performs ~189 gradient update steps. 150 epochs yields ~28,350 total updates — sufficient for convergence on this dataset scale while remaining computationally tractable on CPU.

---

### 2.10 Training Loss Visualisation

**What it does:** After training completes, `matplotlib.pyplot` is used to plot training loss and validation loss curves over 150 epochs side-by-side.

**Technical mechanism:**

```python
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
```

**Why this matters:** Divergence between training loss and validation loss — where training loss continues to decrease while validation loss stagnates or increases — is the primary visual diagnostic for overfitting. This plot is the go-to tool for deciding whether to adjust dropout rates, reduce network capacity, or implement early stopping.

---

### 2.11 Artefact Serialisation & Persistence

**What it does:** All inference-time dependencies are serialised to disk after training in two formats:

1. **Keras native format** (`battery_life_model.h5`) — the full TensorFlow/Keras model including architecture, weights, and optimiser state, saved via `model.save()`.
2. **Pickle format** (`battery_life_model.pkl`, `scaler.pkl`, `label_encoder.pkl`) — Python object serialisation using the `pickle` module.

**Technical mechanism:**

```python
# Keras native save (recommended for model)
model.save("battery_life_model.h5")

# Pickle save (for sklearn preprocessing objects)
with open('battery_life_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

**Important note on `app.py`:** The Streamlit app calls `load_model("battery_life_model.h5")` (Keras native loader) for the neural network, while loading `scaler.pkl` and `label_encoder.pkl` via `pickle.load()`. The `.pkl` version of the model (`battery_life_model.pkl`) is an alternative serialisation format not consumed by the current app.

**Why two formats:** Keras's native `.h5` format is the recommended, TensorFlow-version-portable serialisation for neural network models. Pickle is used for sklearn objects because they lack a built-in serialisation API equivalent to `model.save()`. The dual-format model save provides a fallback.

---

### 2.12 Prediction Inference Function

**What it does:** A standalone `predict_battery_life()` function encapsulates the complete inference pipeline — encoding, scaling, and prediction — as a single callable that accepts raw user-facing inputs and returns the model's scalar prediction.

**Technical mechanism:**

```python
def predict_battery_life(type_discharge, Capacity, Re, Rct,
                          label_encoder, scaler, model):
    # Step 1: Encode categorical input
    type_encoded = label_encoder.transform([type_discharge])[0]

    # Step 2: Assemble feature vector in training-column order
    X_input = np.array([[type_encoded, Capacity, Re, Rct]])

    # Step 3: Apply the fitted scaler (transform only — never re-fit)
    X_scaled = scaler.transform(X_input)

    # Step 4: Run forward pass
    prediction = model.predict(X_scaled)

    return prediction[0]
```

**Why isolate this function:** Encapsulating the preprocessing + inference chain in a single function enforces consistent column ordering and prevents callers from accidentally bypassing a preprocessing step. This function is reused verbatim in both the notebook (for testing) and `app.py` (for production inference).

---

### 2.13 Streamlit Web Application (app.py)

**What it does:** `app.py` implements an interactive browser-based prediction interface using Streamlit. It loads all three serialised artefacts at startup and exposes a form with four input controls, a prediction trigger button, and a result display.

**Technical mechanism:** Streamlit's execution model re-runs the entire script from top to bottom on every user interaction. Model loading occurs unconditionally at module-level (outside any function), meaning artefacts are loaded once per session start and remain in memory for the session lifetime. This is functionally equivalent to a singleton loading pattern within Streamlit's single-threaded execution context.

**UI components:**

| Widget | Type | Input Field | Options / Range |
|---|---|---|---|
| `st.selectbox` | Dropdown | Discharge Type | `charge`, `discharge`, `impedance` |
| `st.number_input` | Numeric | Capacity | min=0.0 |
| `st.number_input` | Numeric | Re | min/max ±1×10¹² |
| `st.number_input` | Numeric | Rct | min/max ±1×10¹² |
| `st.button` | Click trigger | Predict | — |
| `st.write` | Output display | Result | Formatted float string |

**Why Streamlit:** Streamlit converts Python scripts into reactive web apps with minimal boilerplate. For a data science prototype, it eliminates the need for a separate frontend framework (React, Vue) and API layer (Flask, FastAPI), enabling the entire serving stack to be a single 40-line Python file.

---

## 🛠️ Tech Stack & System Architecture

### 3.1 Technology Stack

| Category | Technology | Version | Role & Justification |
|---|---|---|---|
| **Language** | Python | ≥3.9 | Primary development language; ecosystem dominance in ML/data science. |
| **Deep Learning** | TensorFlow / Keras | 2.x | High-level Sequential API for rapid ANN prototyping; GPU-ready without code changes. |
| **ML Preprocessing** | scikit-learn | ≥1.0 | `MinMaxScaler` and `LabelEncoder` for production-grade, pipeline-safe preprocessing. |
| **Data Manipulation** | Pandas | ≥1.3 | DataFrame-native ingestion, missing-value handling, and feature engineering. |
| **Numerical Computing** | NumPy | ≥1.21 | Feature vector construction and array manipulation for the inference function. |
| **Visualisation** | Matplotlib | ≥3.4 | Training/validation loss curve plotting in the notebook. |
| **Web Application** | Streamlit | ≥1.0 | Single-file reactive web app framework for the inference UI. |
| **Serialisation** | pickle (stdlib) | Built-in | Serialisation of sklearn preprocessing objects (scaler, encoder). |
| **Model Persistence** | Keras HDF5 (`.h5`) | TF 2.x | TensorFlow-native model serialisation format preserving architecture + weights. |
| **Dataset** | NASA Battery CSV | — | `metadata.csv` — 7,565-row electrochemical measurement dataset. |
| **Notebook Runtime** | Jupyter Notebook | ≥6.0 | Interactive training, EDA, and visualisation environment. |

---

### 3.2 Repository Directory Structure

```
EV_Battery_Life_Prediction_Using_ANN_Keras-main/
│
├── EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb  # Master training & EDA notebook.
│                                                     # Contains all preprocessing, model
│                                                     # definition, training, evaluation,
│                                                     # and artefact serialisation steps.
│
├── app.py                                            # Streamlit prediction web application.
│                                                     # Loads model + scaler + encoder,
│                                                     # exposes a 4-field form, runs inference,
│                                                     # and displays the predicted temperature.
│
├── metadata.csv                                      # Primary dataset — 7,565 rows × 10 cols.
│                                                     # NASA Battery Dataset (charge/discharge/
│                                                     # impedance cycles for 34 Li-ion cells).
│
├── battery_life_model.pkl                            # Pickle-serialised Keras model object.
│                                                     # Alternative format; app.py uses .h5 instead.
│                                                     # Size: ~59 KB.
│
├── battery_life_model.h5                             # [Generated by training] TensorFlow/Keras
│                                                     # native HDF5 model file. Consumed by app.py
│                                                     # via tf.keras.models.load_model(). NOT
│                                                     # included in ZIP — must be regenerated by
│                                                     # running the notebook.
│
├── scaler.pkl                                        # Pickle-serialised MinMaxScaler fitted on
│                                                     # X_train. Must be loaded and applied at
│                                                     # inference time. Size: ~757 bytes.
│
├── label_encoder.pkl                                 # Pickle-serialised LabelEncoder fitted on
│                                                     # the 'type' column. Maps charge/discharge/
│                                                     # impedance → 0/1/2. Size: ~275 bytes.
│
└── README.md                                         # Project documentation (this file).
```

---

### 3.3 System Architecture & Data Flow

The system comprises two distinct operational phases: an offline **Training Pipeline** and an online **Inference Pipeline**.

#### Training Pipeline (Notebook)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
│                    (Jupyter Notebook — Offline)                             │
└─────────────────────────────────────────────────────────────────────────────┘

  metadata.csv
       │
       ▼
┌─────────────────────┐
│   Data Ingestion    │  pd.read_csv() → DataFrame (7,565 × 10)
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Column Pruning     │  Drop: start_time, battery_id, test_id, uid, filename
└─────────────────────┘  Remaining: type, ambient_temperature, Capacity, Re, Rct
       │
       ▼
┌─────────────────────┐
│  Type Coercion      │  pd.to_numeric(errors='coerce') on Capacity, Re, Rct
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Mean Imputation    │  fillna(column.mean()) for Capacity, Re, Rct
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Label Encoding     │  LabelEncoder.fit_transform(df['type'])
│  (fit_transform)    │  charge→0, discharge→1, impedance→2
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Feature / Target   │  X = [type, Capacity, Re, Rct]
│  Separation         │  y = ambient_temperature
└─────────────────────┘
       │
       ▼
┌─────────────────────┐
│  Train/Test Split   │  80% train (6,052 rows) / 20% test (1,513 rows)
│  random_state=42    │
└─────────────────────┘
       │
       ├──────────── X_train ──────────────────────────────────────┐
       │                                                           │
       ▼                                                           ▼
┌─────────────────────┐                                  ┌─────────────────────┐
│  MinMaxScaler       │  fit_transform(X_train)          │  MinMaxScaler       │
│  (fit on X_train)   │──────────────────────────────▶   │  transform(X_test)  │
└─────────────────────┘                                  └─────────────────────┘
       │  X_train_scaled                                        │  X_test_scaled
       ▼                                                        │
┌─────────────────────┐                                         │
│  ANN Training       │  model.fit(X_train_scaled, y_train,     │
│  Sequential MLP     │    validation_data=(X_test_scaled,      │
│  64→32→1 neurons    │    y_test), epochs=150, batch_size=32)  │
└─────────────────────┘                                         │
       │                                                        │
       ▼                                                        ▼
┌─────────────────────┐                            ┌────────────────────────┐
│  Model Evaluation   │  model.evaluate() → MSE   │  Loss Visualisation    │
│                     │  on X_test_scaled          │  Matplotlib curve plot │
└─────────────────────┘                            └────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ARTEFACT SERIALISATION                     │
│  model.save("battery_life_model.h5")                        │
│  pickle.dump(model)    → battery_life_model.pkl             │
│  pickle.dump(scaler)   → scaler.pkl                         │
│  pickle.dump(encoder)  → label_encoder.pkl                  │
└─────────────────────────────────────────────────────────────┘
```

#### Inference Pipeline (Streamlit App)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                                  │
│                      (Streamlit App — Online)                               │
└─────────────────────────────────────────────────────────────────────────────┘

  App Startup
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Artefact Loading (once per session)                     │
│  load_model("battery_life_model.h5") → model             │
│  pickle.load("scaler.pkl")           → scaler            │
│  pickle.load("label_encoder.pkl")    → label_encoder     │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Streamlit UI Render            │
│  ┌─────────────────────────┐    │
│  │ Discharge Type Dropdown │    │   User selects: charge | discharge | impedance
│  │ Capacity Number Input   │    │   User enters: e.g., 1.6743
│  │ Re Number Input         │    │   User enters: e.g., 0.056
│  │ Rct Number Input        │    │   User enters: e.g., 0.201
│  │ [Predict Battery Life]  │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
       │  (on button click)
       ▼
┌─────────────────────────────────┐
│  Label Encoding (inference)     │  label_encoder.transform([type_discharge])
│  → integer (0, 1, or 2)         │  (transform only — encoder already fitted)
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Feature Vector Assembly        │  np.array([[type_encoded, Capacity, Re, Rct]])
│  Shape: (1, 4)                  │  Column order MUST match training order
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  MinMaxScaler Transform         │  scaler.transform(X_input)
│  (transform only — never fit)   │  Applies training-fitted bounds
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  ANN Forward Pass               │  model.predict(X_input_scaled)
│  4 → 64 → 32 → 1               │  Returns shape (1, 1) array
└─────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Result Display                 │  st.write(f"Predicted battery life: {pred[0]}")
└─────────────────────────────────┘
```

---

## 📊 Dataset Schema & Data Models

### 4.1 Primary Dataset: `metadata.csv`

The dataset contains **7,565 rows × 10 columns** of battery cycle measurements from NASA's Prognostics Center of Excellence. Each row records metadata and aggregate measurements for one test event associated with a specific battery cell.

| Field Name | Data Type | Constraints | Description & Purpose |
|---|---|---|---|
| `type` | String (Categorical) | NOT NULL; Values: `charge`, `discharge`, `impedance` | Identifies the nature of the battery test cycle. Charge cycles apply current to replenish capacity. Discharge cycles draw current from the battery. Impedance cycles measure internal resistance via electrochemical impedance spectroscopy (EIS). |
| `start_time` | String (Array) | NOT NULL | Timestamp of test start, stored as a Python list-formatted string (e.g., `[2010. 7. 21. 15. 0. 35.093]` representing year, month, day, hour, minute, second). Dropped before modelling due to non-standard format. |
| `ambient_temperature` | Integer | NOT NULL | Temperature (°C) of the surrounding environment during the test. Takes values: 4°C, 24°C, or 44°C in this dataset. **This is the regression target variable.** |
| `battery_id` | String (Categorical) | NOT NULL; 34 unique values | Identifier for the physical battery cell (e.g., `B0025`, `B0047`). 34 distinct cells in the dataset. Dropped before modelling. |
| `test_id` | Integer | NOT NULL; Range: 0–615 | Sequential test number for a given battery cell. Monotonically increases over the battery's lifetime. Dropped before modelling. |
| `uid` | Integer | NOT NULL; Range: 1–7565; Unique | Global unique record identifier across all batteries and tests. Dropped before modelling. |
| `filename` | String | NOT NULL | Original source CSV file from the NASA dataset archive from which this row was derived (e.g., `00001.csv`). Dropped before modelling. |
| `Capacity` | String → Float (coerced) | Nullable (~63% missing); Valid range: 0.0–2.0 Ah approx. | Battery capacity in ampere-hours measured during a discharge cycle. Populated only for `type == 'discharge'` rows. Present in ~2,794 of 7,565 rows. **Key predictive feature.** Missing values filled with column mean (~1.55 Ah) during preprocessing. |
| `Re` | String → Float (coerced) | Nullable (~74% missing); Valid range: 0.04–0.11 Ω approx. | Estimated electrolyte resistance (Ohms) derived from impedance spectroscopy. Populated only for `type == 'impedance'` rows. Present in ~1,956 rows. **Key predictive feature.** Missing values filled with column mean during preprocessing. |
| `Rct` | String → Float (coerced) | Nullable (~74% missing); Valid range: 0.05–2.0 Ω approx. | Estimated charge-transfer resistance (Ohms) derived from impedance spectroscopy. Populated only for `type == 'impedance'` rows. Present in ~1,956 rows. **Key predictive feature.** Missing values filled with column mean during preprocessing. |

---

### 4.2 Cycle Type Distribution

| Cycle Type | Row Count | % of Dataset | Features Populated |
|---|---|---|---|
| `charge` | 2,815 | 37.2% | type, start_time, ambient_temperature, battery_id, test_id, uid, filename |
| `discharge` | 2,794 | 36.9% | All columns above + **Capacity** |
| `impedance` | 1,956 | 25.9% | All columns above + **Re** + **Rct** |

---

### 4.3 Battery Cell Population

| Attribute | Value |
|---|---|
| Total distinct battery cells | 34 |
| Battery ID range | B0025 – B0056 (non-contiguous; some IDs absent) |
| Tests per cell | Varies from ~60 to ~615 (depends on cell endurance) |
| Ambient temperatures tested | 4°C, 24°C, 44°C |

---

### 4.4 Model Input Feature Schema (Post-Preprocessing)

This is the schema of the four-column feature matrix `X` passed to the ANN after all preprocessing:

| Feature Name | Encoded From | Data Type | MinMax Range (approx.) | Meaning |
|---|---|---|---|---|
| `type` (encoded) | `type` (string) | Integer: 0, 1, or 2 | [0, 2] → scaled [0.0, 1.0] | 0=charge, 1=discharge, 2=impedance |
| `Capacity` | `Capacity` (string→float) | Float | [~0.4, ~2.0] Ah → [0.0, 1.0] | Current deliverable charge capacity of the cell |
| `Re` | `Re` (string→float) | Float | [~0.04, ~0.11] Ω → [0.0, 1.0] | Electrolyte internal resistance |
| `Rct` | `Rct` (string→float) | Float | [~0.05, ~2.0] Ω → [0.0, 1.0] | Charge-transfer resistance at electrode surface |

---

### 4.5 Serialised Artefact Registry

| Artefact File | Object Type | Serialisation Format | Size on Disk | Loaded By |
|---|---|---|---|---|
| `battery_life_model.h5` | `tf.keras.Sequential` | Keras HDF5 | ~200–400 KB (estimated) | `app.py` via `load_model()` |
| `battery_life_model.pkl` | `tf.keras.Sequential` | Python pickle | ~59 KB | Not consumed by app; alternative format |
| `scaler.pkl` | `sklearn.MinMaxScaler` | Python pickle | ~757 bytes | `app.py` via `pickle.load()` |
| `label_encoder.pkl` | `sklearn.LabelEncoder` | Python pickle | ~275 bytes | `app.py` via `pickle.load()` |

---

## 🌐 API Reference & Integration Guide

### 5.1 Overview

This project does not expose a REST API in its current form. The inference interface is delivered entirely through the Streamlit web application (`app.py`). The following section documents the **public interface** of the core inference function, the Streamlit application's interaction contract, and provides curl-equivalent integration notes for teams wishing to extend the system with a REST layer.

---

### 5.2 Core Inference Function Interface

**Function:** `predict_battery_life()`

**Location:** `app.py` (lines 18–30); also defined in notebook Cell 20.

**Signature:**

```python
def predict_battery_life(
    type_discharge: str,         # One of: 'charge', 'discharge', 'impedance'
    Capacity: float,             # Battery capacity in Ah (e.g., 1.6743)
    Re: float,                   # Electrolyte resistance in Ohms (e.g., 0.056)
    Rct: float,                  # Charge-transfer resistance in Ohms (e.g., 0.201)
    label_encoder: LabelEncoder, # Fitted sklearn LabelEncoder (loaded from label_encoder.pkl)
    scaler: MinMaxScaler,        # Fitted sklearn MinMaxScaler (loaded from scaler.pkl)
    model: Sequential            # Loaded Keras Sequential model
) -> np.ndarray                  # Shape: (1,) — predicted ambient_temperature in °C
```

**Input Constraints:**

| Parameter | Accepted Values | Notes |
|---|---|---|
| `type_discharge` | `'charge'`, `'discharge'`, `'impedance'` | Case-sensitive. Any other string raises `ValueError` from LabelEncoder. |
| `Capacity` | Float, ≥ 0.0 | Typical range 0.4–2.0 Ah for Li-ion cells. Values outside training distribution extrapolate. |
| `Re` | Float, any finite value | Typically 0.04–0.11 Ω. App UI accepts ±1×10¹². |
| `Rct` | Float, any finite value | Typically 0.05–2.0 Ω. App UI accepts ±1×10¹². |

**Return Value:**

A NumPy array of shape `(1,)` containing the predicted `ambient_temperature` in degrees Celsius. The model output is continuous and may not align exactly with the three discrete temperatures in the training set (4, 24, 44°C).

**Example call (notebook/Python):**

```python
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load artefacts
model         = load_model("battery_life_model.h5")
scaler        = pickle.load(open("scaler.pkl",        "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Run inference
result = predict_battery_life(
    type_discharge = 'discharge',
    Capacity       = 1.674305,
    Re             = 0.056058,
    Rct            = 0.200970,
    label_encoder  = label_encoder,
    scaler         = scaler,
    model          = model
)

print(f"Predicted ambient temperature: {result[0]:.2f} °C")
# Example output: Predicted ambient temperature: 23.87 °C
```

**Error cases:**

| Error Condition | Exception Raised | Root Cause |
|---|---|---|
| `type_discharge='fast_charge'` | `sklearn.exceptions.NotFittedError` or `ValueError` | LabelEncoder encounters an unseen class not in `['charge', 'discharge', 'impedance']`. |
| `Capacity=None` or `np.nan` | `ValueError` in `scaler.transform()` | MinMaxScaler cannot process NaN values. |
| Missing `.h5` model file | `OSError` / `FileNotFoundError` | `load_model()` path does not exist. Run the notebook to regenerate. |
| Missing `.pkl` files | `FileNotFoundError` | Scaler/encoder pickles not found. Run the notebook to regenerate. |

---

### 5.3 Streamlit Application Interaction Contract

The Streamlit app does not expose HTTP endpoints. Interaction occurs through the browser UI at `http://localhost:8501`.

**Input Form Fields:**

| UI Control | Widget Type | Underlying Input | Default Value | Accepted Range |
|---|---|---|---|---|
| Discharge Type | `st.selectbox` | `type_discharge: str` | `'charge'` (first option) | `'charge'` \| `'discharge'` \| `'impedance'` |
| Capacity | `st.number_input` | `Capacity: float` | `0.0` | `[0.0, ∞)` |
| Re | `st.number_input` | `Re: float` | `0.0` | `[-1×10¹², 1×10¹²]` |
| Rct | `st.number_input` | `Rct: float` | `0.0` | `[-1×10¹², 1×10¹²]` |

**Trigger:** `st.button("Predict Battery Life")` — clicking this button causes Streamlit to re-run the script, executing the `predict_battery_life()` call with the current widget values.

**Output:** `st.write(f"The predicted battery life is: {predicted_battery_life} units")` — displayed below the button. Note: "units" here refers to the model's prediction, which is the ambient temperature in degrees Celsius (the label used during training).

---

### 5.4 Extending with a REST API (Flask/FastAPI Reference)

If you require programmatic access for integration with fleet management systems or monitoring dashboards, the following pattern converts the inference function into an HTTP endpoint:

```python
# fastapi_server.py (extension — not included in the repository)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app_api = FastAPI(title="EV Battery Life Prediction API", version="1.0.0")

# Load artefacts once at startup
model_nn      = load_model("battery_life_model.h5")
scaler        = pickle.load(open("scaler.pkl",        "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

class PredictionRequest(BaseModel):
    type_discharge: str   = Field(..., example="discharge",
                                  description="One of: charge, discharge, impedance")
    capacity:       float = Field(..., example=1.6743, ge=0.0,
                                  description="Battery capacity in Ah")
    re:             float = Field(..., example=0.0561,
                                  description="Electrolyte resistance (Ohms)")
    rct:            float = Field(..., example=0.2010,
                                  description="Charge-transfer resistance (Ohms)")

class PredictionResponse(BaseModel):
    predicted_ambient_temperature_celsius: float
    input_received: dict

@app_api.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    summary="Predict ambient battery temperature",
    tags=["Inference"]
)
def predict(request: PredictionRequest):
    """
    Accepts four electrochemical features and returns the predicted
    ambient operating temperature of the EV battery cell in Celsius.
    """
    if request.type_discharge not in ['charge', 'discharge', 'impedance']:
        raise HTTPException(
            status_code=422,
            detail=f"type_discharge must be one of 'charge', 'discharge', 'impedance'. "
                   f"Got: '{request.type_discharge}'"
        )
    try:
        encoded = label_encoder.transform([request.type_discharge])[0]
        X = np.array([[encoded, request.capacity, request.re, request.rct]])
        X_scaled = scaler.transform(X)
        prediction = model_nn.predict(X_scaled)
        return PredictionResponse(
            predicted_ambient_temperature_celsius=float(prediction[0][0]),
            input_received=request.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Sample POST Request:**

```http
POST /api/v1/predict HTTP/1.1
Host: localhost:8000
Content-Type: application/json

{
  "type_discharge": "discharge",
  "capacity": 1.6743047446975208,
  "re": 0.05605783343888099,
  "rct": 0.20097016584458333
}
```

**Sample Success Response (200 OK):**

```json
{
  "predicted_ambient_temperature_celsius": 23.87,
  "input_received": {
    "type_discharge": "discharge",
    "capacity": 1.6743047446975208,
    "re": 0.05605783343888099,
    "rct": 0.20097016584458333
  }
}
```

**Sample Error Response (422 Unprocessable Entity):**

```json
{
  "detail": "type_discharge must be one of 'charge', 'discharge', 'impedance'. Got: 'fast_charge'"
}
```

**Sample Error Response (500 Internal Server Error):**

```json
{
  "detail": "Input contains NaN, infinity or a value too large for dtype('float64')."
}
```

---

## ⚙️ Installation, Configuration & Environment

### 6.1 Prerequisites

Ensure the following are installed on your system before proceeding:

| Requirement | Minimum Version | Installation Reference |
|---|---|---|
| Python | 3.9 | https://www.python.org/downloads/ |
| pip | 21.0 | Included with Python 3.9+ |
| Git | 2.30 | https://git-scm.com/downloads |
| (Optional) CUDA Toolkit | 11.2 | Required only for GPU-accelerated training |
| (Optional) cuDNN | 8.1 | Required only for GPU-accelerated training |

> **Note on GPU Training:** The model is small (2,433 parameters) and trains comfortably on CPU in a few minutes. GPU acceleration is unnecessary for this dataset scale.

---

### 6.2 Environment Variables

This project does not use environment variables or `.env` files in its current form — all configuration is either hardcoded in the notebook (file paths, hyperparameters) or exposed via the Streamlit UI (inference inputs). If you extend this project to add a database backend, cloud storage, or an authenticated REST API, the following `.env` template is recommended:

```bash
# .env.example — Environment variable template for extended deployments
# Copy to .env and fill in your values. Never commit .env to version control.

# ── Model & Artefact Paths ─────────────────────────────────────────────────
# Paths to serialised model and preprocessing artefacts.
# Override if artefacts are stored outside the working directory.
MODEL_PATH=./battery_life_model.h5
SCALER_PATH=./scaler.pkl
LABEL_ENCODER_PATH=./label_encoder.pkl

# ── Streamlit Configuration ────────────────────────────────────────────────
# Port on which the Streamlit server listens.
STREAMLIT_SERVER_PORT=8501
# Set to false to disable Streamlit's browser auto-open on launch.
STREAMLIT_SERVER_HEADLESS=false
# Set to true in production/containerised environments.
# STREAMLIT_SERVER_HEADLESS=true

# ── Training Hyperparameters ───────────────────────────────────────────────
# (Used if you refactor training into a Python script with argparse)
TRAINING_EPOCHS=150
BATCH_SIZE=32
TEST_SIZE=0.2
RANDOM_STATE=42
DROPOUT_RATE=0.2

# ── Dataset Configuration ──────────────────────────────────────────────────
DATASET_PATH=./metadata.csv

# ── Logging ────────────────────────────────────────────────────────────────
# Python logging level: DEBUG | INFO | WARNING | ERROR | CRITICAL
LOG_LEVEL=INFO

# ── Extended: REST API Configuration (if FastAPI layer added) ──────────────
# API_HOST=0.0.0.0
# API_PORT=8000
# API_WORKERS=4

# ── Extended: Cloud Storage (if artefacts stored remotely) ────────────────
# AWS_ACCESS_KEY_ID=your_access_key_here
# AWS_SECRET_ACCESS_KEY=your_secret_key_here
# AWS_DEFAULT_REGION=eu-central-1
# S3_MODEL_BUCKET=ev-battery-models
# S3_MODEL_KEY=battery_life_model.h5
```

---

### 6.3 Step-by-Step Local Setup

Follow these steps in order to clone, configure, and run the project locally.

**Step 1 — Clone the repository:**

```bash
git clone https://github.com/wittyswayam/EV_Battery_Life_Prediction_Using_ANN_Keras.git
cd EV_Battery_Life_Prediction_Using_ANN_Keras
```

**Step 2 — Create and activate a Python virtual environment** (strongly recommended to avoid dependency conflicts):

```bash
# Create virtual environment
python3 -m venv .venv

# Activate (Linux / macOS)
source .venv/bin/activate

# Activate (Windows CMD)
.venv\Scripts\activate.bat

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

**Step 3 — Install all required dependencies:**

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

Or, if you create a `requirements.txt` (see Section 8.1), install via:

```bash
pip install -r requirements.txt
```

**Step 4 — Verify the dataset is present:**

```bash
ls -lh metadata.csv
# Expected output: -rw-r--r-- 1 user group 849K <date> metadata.csv
```

**Step 5 — Train the model by running the Jupyter Notebook** (generates `battery_life_model.h5`):

```bash
# Start Jupyter
jupyter notebook

# In the browser, open:
# EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb

# Run all cells: Kernel → Restart & Run All
# Training takes approximately 2–8 minutes on CPU depending on hardware.
# Upon completion, the following files are written to the current directory:
#   battery_life_model.h5
#   battery_life_model.pkl
#   scaler.pkl
#   label_encoder.pkl
```

> **Important:** `battery_life_model.h5` is NOT included in the repository ZIP because it is a binary artefact generated at training time. You MUST run the notebook before launching the Streamlit app, or `app.py` will raise `FileNotFoundError` at startup.

**Step 6 — Verify all artefacts exist:**

```bash
ls -lh battery_life_model.h5 battery_life_model.pkl scaler.pkl label_encoder.pkl
# All four files should be present.
```

**Step 7 — Launch the Streamlit prediction app:**

```bash
streamlit run app.py
```

Streamlit will print output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL:  http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Navigate to `http://localhost:8501` in your browser. The app loads all artefacts and renders the prediction form.

**Step 8 — Make a test prediction:**

In the app:
1. Select **Discharge Type:** `discharge`
2. Enter **Capacity:** `1.6743`
3. Enter **Re:** `0.056058`
4. Enter **Rct:** `0.200970`
5. Click **Predict Battery Life**

Expected output: A predicted value close to 4.0 or 24.0 (the discrete ambient temperatures in the dataset).

---

## 🧪 Running Tests, Linting & CI/CD

### 7.1 Unit Tests

The repository does not currently include a formal test suite. The following test structure is recommended for contributors extending the project:

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Create test file at tests/test_inference.py (see below)
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Recommended test file (`tests/test_inference.py`):**

```python
import pickle
import numpy as np
import pytest
from tensorflow.keras.models import load_model

# Fixtures
@pytest.fixture(scope="module")
def artefacts():
    model         = load_model("battery_life_model.h5")
    scaler        = pickle.load(open("scaler.pkl",        "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, label_encoder

def test_label_encoder_classes(artefacts):
    _, _, le = artefacts
    assert set(le.classes_) == {'charge', 'discharge', 'impedance'}

def test_scaler_feature_count(artefacts):
    _, scaler, _ = artefacts
    assert scaler.n_features_in_ == 4

def test_model_output_shape(artefacts):
    model, scaler, le = artefacts
    from app import predict_battery_life
    result = predict_battery_life('discharge', 1.674, 0.056, 0.201, le, scaler, model)
    assert result.shape == (1,)

def test_prediction_range(artefacts):
    """Predicted temperature should be within a reasonable physical range."""
    model, scaler, le = artefacts
    from app import predict_battery_life
    result = predict_battery_life('discharge', 1.674, 0.056, 0.201, le, scaler, model)
    assert -20.0 <= float(result[0]) <= 80.0, "Prediction outside physically plausible range"

def test_unknown_type_raises(artefacts):
    model, scaler, le = artefacts
    from app import predict_battery_life
    with pytest.raises(ValueError):
        predict_battery_life('fast_charge', 1.674, 0.056, 0.201, le, scaler, model)
```

---

### 7.2 Code Quality & Linting

```bash
# Install linting tools
pip install flake8 black isort

# PEP 8 compliance check
flake8 app.py --max-line-length=120

# Auto-format with Black
black app.py

# Sort imports
isort app.py

# Type checking (optional, requires stubs)
pip install mypy
mypy app.py --ignore-missing-imports
```

---

### 7.3 CI/CD with GitHub Actions

The repository does not currently include a GitHub Actions workflow. The following workflow file (`/.github/workflows/ci.yml`) is recommended:

```yaml
# .github/workflows/ci.yml
name: CI — Lint & Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install tensorflow scikit-learn pandas numpy matplotlib streamlit \
                      pytest pytest-cov flake8 black

      - name: Lint with flake8
        run: flake8 app.py --max-line-length=120 --ignore=E501,W503

      - name: Check formatting with Black
        run: black app.py --check

      - name: Run notebook (training + artefact generation)
        run: |
          pip install nbconvert
          jupyter nbconvert --to notebook --execute \
            EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb \
            --output executed_notebook.ipynb \
            --ExecutePreprocessor.timeout=600

      - name: Run tests with coverage
        run: pytest tests/ -v --cov=. --cov-report=xml

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

---

## 🚀 Production Deployment & Best Practices

### 8.1 Requirements File

Create a `requirements.txt` for reproducible dependency installation:

```text
# requirements.txt — generated for EV Battery Life Prediction
tensorflow>=2.10,<3.0
scikit-learn>=1.0,<2.0
pandas>=1.3,<3.0
numpy>=1.21,<2.0
matplotlib>=3.4,<4.0
streamlit>=1.10,<2.0
jupyter>=1.0
ipykernel>=6.0
```

---

### 8.2 Docker Containerisation

The following Dockerfile packages the Streamlit app and all artefacts into a portable container image:

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TensorFlow (slim image may lack some)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and all artefacts
# Note: battery_life_model.h5 must be generated by running the notebook BEFORE building.
COPY app.py .
COPY battery_life_model.h5 .
COPY scaler.pkl .
COPY label_encoder.pkl .

# Streamlit configuration for headless production mode
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build and run:**

```bash
# Step 1: Run the notebook to generate battery_life_model.h5 (if not already present)
jupyter nbconvert --to notebook --execute \
  EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb \
  --ExecutePreprocessor.timeout=600

# Step 2: Build the Docker image
docker build -t ev-battery-prediction:latest .

# Step 3: Run the container
docker run -d \
  --name ev-battery-app \
  -p 8501:8501 \
  ev-battery-prediction:latest

# Step 4: Access the app
# Open http://localhost:8501 in your browser.

# Step 5: View logs
docker logs -f ev-battery-app
```

**Docker Compose (for multi-service deployments):**

```yaml
# docker-compose.yml
version: "3.9"
services:
  ev-prediction-app:
    build: .
    image: ev-battery-prediction:latest
    container_name: ev-battery-app
    ports:
      - "8501:8501"
    restart: unless-stopped
    environment:
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_PORT=8501
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

```bash
# Start with docker-compose
docker-compose up -d

# Stop
docker-compose down
```

---

### 8.3 Deploying to Streamlit Community Cloud

For public deployments without infrastructure management:

1. Push the repository (including `metadata.csv`, `scaler.pkl`, `label_encoder.pkl`, and `battery_life_model.h5`) to GitHub.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in.
3. Click **"New app"**, select your repository and branch, and set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud installs dependencies automatically from your repository.

> **Note:** The 1 GB free tier may be tight if `battery_life_model.h5` is large. Use Git LFS for binary artefacts exceeding 100 MB.

---

### 8.4 Performance & Optimisation Considerations

| Area | Current State | Recommended Improvement |
|---|---|---|
| **Artefact Loading** | Loaded on every Streamlit script re-run (Streamlit caches if using `@st.cache_resource`) | Add `@st.cache_resource` decorator to artefact loading to prevent re-loading on each interaction |
| **Model Size** | 2,433 parameters — very lightweight | No optimisation needed; consider TensorFlow Lite conversion for mobile deployment |
| **Batch Inference** | Single-sample inference in the UI | Add CSV upload + batch prediction mode for fleet-scale use cases |
| **Input Validation** | Minimal (Streamlit widget min/max only) | Add server-side validation catching out-of-distribution inputs before model call |
| **Preprocessing Speed** | Instantaneous for single-row inference | No bottleneck at current scale |

**Recommended `@st.cache_resource` pattern for `app.py`:**

```python
import streamlit as st
import pickle
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artefacts():
    """Load all inference artefacts once per session. Cached by Streamlit."""
    model         = load_model("battery_life_model.h5")
    scaler        = pickle.load(open("scaler.pkl",        "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artefacts()
```

This ensures artefacts are loaded only once per server session rather than on every widget interaction, reducing latency from ~1–2 seconds to milliseconds after the initial load.

---

### 8.5 Security Posture

| Security Consideration | Current State | Recommended Action |
|---|---|---|
| **Pickle Deserialisation** | `scaler.pkl` and `label_encoder.pkl` are loaded from disk at startup. Pickle deserialisation of untrusted files is a known RCE vector. | Never load `.pkl` files from untrusted or user-supplied sources. Validate artefact checksums (SHA-256) on startup in production. |
| **Input Validation** | Streamlit `number_input` enforces min/max bounds client-side only. | Add server-side bounds checking and out-of-distribution detection before calling `model.predict()`. |
| **Model File Integrity** | No checksum verification of `battery_life_model.h5`. | Store SHA-256 hashes of all artefacts and verify at load time in production. |
| **CORS & Network** | Streamlit default configuration. | In production, set `STREAMLIT_SERVER_ENABLECORS=false` and place behind a reverse proxy (nginx) with HTTPS. |
| **Dependency Vulnerabilities** | Not audited. | Run `pip audit` regularly and pin all dependency versions in `requirements.txt`. |

---

## 🤝 Contribution, Versioning & License

### 9.1 Contributing Guidelines

Contributions are warmly welcomed. To contribute:

1. **Fork** the repository on GitHub.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/lstm-temporal-model
   ```
3. **Make your changes.** Ensure the notebook still executes end-to-end (`Kernel → Restart & Run All`) and `streamlit run app.py` launches without errors.
4. **Run linting** before committing:
   ```bash
   flake8 app.py --max-line-length=120
   black app.py
   ```
5. **Commit** following the Conventional Commits specification (see Section 9.2).
6. **Push** your branch and **open a Pull Request** against `main` with a clear description of the change, the motivation, and any relevant test results.
7. A maintainer will review and provide feedback within 5 business days.

**Opening Issues:**
- Use the GitHub Issues tab.
- For bug reports, include: Python version, TensorFlow version, OS, full error traceback, and steps to reproduce.
- For feature requests, describe the use case, expected behaviour, and any relevant papers or references.

---

### 9.2 Git Commit Convention

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. All commit messages must conform to:

```
<type>(<optional-scope>): <short description>

[optional body]

[optional footer]
```

**Allowed types:**

| Type | When to Use |
|---|---|
| `feat` | A new feature or capability (e.g., `feat(app): add batch CSV prediction mode`) |
| `fix` | A bug fix (e.g., `fix(app): correct column order in feature vector`) |
| `docs` | Documentation changes only (e.g., `docs: add FastAPI integration example`) |
| `refactor` | Code restructuring with no behaviour change (e.g., `refactor: extract artefact loading into separate module`) |
| `perf` | Performance improvement (e.g., `perf(app): add st.cache_resource to artefact loading`) |
| `test` | Adding or modifying tests (e.g., `test: add unit tests for predict_battery_life function`) |
| `chore` | Maintenance tasks — dependency updates, CI changes (e.g., `chore: pin TensorFlow to 2.12`) |
| `style` | Code style changes — formatting, whitespace (e.g., `style: apply Black formatting to app.py`) |

**Examples:**

```bash
git commit -m "feat(model): add LSTM architecture for temporal discharge sequence modelling"
git commit -m "fix(scaler): replace StandardScaler reference in app.py with MinMaxScaler"
git commit -m "docs: document FastAPI REST extension pattern in README"
git commit -m "perf(app): cache model artefacts with st.cache_resource"
```

---

### 9.3 Roadmap & Future Improvements

| Priority | Improvement | Description |
|---|---|---|
| High | `@st.cache_resource` for artefact loading | Prevents re-loading model on every Streamlit interaction |
| High | Formal test suite (`pytest`) | Unit tests for inference function and preprocessing pipeline |
| High | Temporal train/test split | Split data by time rather than randomly to prevent future data leakage |
| Medium | LSTM/GRU model | Exploit sequential structure of discharge cycles for improved accuracy |
| Medium | State of Health (SOH) target | Derive SOH = Capacity / InitialCapacity per battery_id and predict it directly |

---

### 9.5 Dataset Attribution

The dataset (`metadata.csv`) is derived from the **NASA Prognostics Center of Excellence (PCoE) Battery Dataset**:

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
> URL: [https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
