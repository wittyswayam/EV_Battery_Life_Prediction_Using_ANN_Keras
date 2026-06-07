# EV_Battery_Life_Prediction_Using_ANN_Keras

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-f7931e?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Dataset: NASA](https://img.shields.io/badge/Dataset-NASA_PCoE_Battery-blueviolet?style=flat-square)](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **Author:** [wittyswayam](https://github.com/wittyswayam)

---

## What This Project Does

This project trains a feed-forward Artificial Neural Network on the NASA Prognostics Center of Excellence (PCoE) Battery Dataset and serves it through a Streamlit web interface. Given four electrochemical measurements from a battery cycle — cycle type, capacity, electrolyte resistance (Re), and charge-transfer resistance (Rct) — the model predicts the ambient temperature at which that cycle occurred.

The dataset covers 7,565 charge, discharge, and impedance spectroscopy cycles across 34 Li-ion cells (B0025–B0056), tested at 4 °C, 24 °C, and 44 °C. Predicting temperature from electrochemical data alone is useful in battery degradation modelling pipelines where sensor logs may be incomplete or where you want a software-only cross-check against recorded temperatures.

---

## Project Structure

```
EV_Battery_Life_Prediction_Using_ANN_Keras-main/
│
├── app/                          # Application source package
│   ├── config.py                 # Centralized configuration (paths, constants)
│   ├── logger.py                 # Logging setup (stdout + rotating file)
│   ├── predictor.py              # PredictionService: artifact loading + inference
│   ├── validators.py             # Input validation, PredictionInput dataclass
│   └── ui/
│       ├── components.py         # Reusable Streamlit UI components
│       └── pages/
│           └── predict.py        # Prediction page view
│
├── artifacts/                    # Trained model artifacts (gitignored for .h5)
│   ├── battery_life_model.pkl    # Pickled Keras model
│   ├── scaler.pkl                # Fitted MinMaxScaler
│   └── label_encoder.pkl         # Fitted LabelEncoder
│
├── data/
│   └── metadata.csv              # NASA PCoE Battery dataset (7,565 rows)
│
├── notebooks/
│   └── EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb
│
├── tests/
│   ├── test_validators.py        # Validation logic unit tests (no artifacts needed)
│   └── test_predictor.py         # PredictionService integration tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions: lint, format check, tests
│
├── main.py                       # Streamlit entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml                # black, isort, pytest, coverage config
├── .flake8
├── .env.example
├── .gitignore
└── README.md
```

---

## Architecture

The project is split into two clearly separated concerns: offline training and online inference. They share nothing at runtime except the serialized artifacts written to the `artifacts/` directory.

### Training Pipeline (Notebook)

All training happens in `notebooks/EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb`. The pipeline steps are:

1. **Ingestion** — `pd.read_csv()` on `data/metadata.csv`. Numeric columns with non-numeric sentinel strings (an artifact of how the NASA dataset was aggregated) are coerced with `pd.to_numeric(errors='coerce')`.

2. **Column pruning** — Five columns that encode provenance rather than battery physics (`start_time`, `battery_id`, `test_id`, `uid`, `filename`) are dropped. Keeping them would let the model memorize cell identities.

3. **Imputation** — `Capacity`, `Re`, and `Rct` carry structural missingness by design: each is only populated for one cycle type. Missing values are filled with the column mean. This is a known limitation (see section below).

4. **Encoding** — The `type` column (`charge`, `discharge`, `impedance`) is ordinally encoded with `sklearn.LabelEncoder` fit on the full dataset. The fitted encoder is saved to `artifacts/label_encoder.pkl`.

5. **Split** — 80/20 random split with `random_state=42`.

6. **Scaling** — `MinMaxScaler` is fit exclusively on `X_train` and applied via `.transform()` to both train and test sets. The fitted scaler is saved to `artifacts/scaler.pkl`. Fitting on the full dataset would leak test-set range information.

7. **Training** — A Sequential Keras model with architecture `4 → Dense(64, ReLU) → Dropout(0.2) → Dense(32, ReLU) → Dropout(0.2) → Dense(1, linear)`. Optimizer: Adam. Loss: MSE. Epochs: 150. Batch size: 32. Validation data passed at each epoch.

8. **Serialization** — The trained model is pickled to `artifacts/battery_life_model.pkl`. If you want to save in Keras HDF5 format instead, use `model.save('artifacts/battery_life_model.h5')`; the inference service supports both formats with minor modification.

### Inference Pipeline (Streamlit App)

`main.py` is the Streamlit entry point. It delegates to `app/ui/pages/predict.py`, which:

- Loads `PredictionService` exactly once per server session via `@st.cache_resource`. Without this, Streamlit reloads all artifacts on every widget interaction, adding 1–2 seconds of latency per click.
- Renders inputs through component functions in `app/ui/components.py`.
- Passes raw user inputs to `app.validators.validate_prediction_input()` before they touch the model.
- Calls `PredictionService.predict()`, which handles encoding, scaling, and inference internally.
- Renders results or error messages depending on the outcome.

The service layer (`app/predictor.py`) is completely decoupled from Streamlit. You can import `PredictionService` in a script, a FastAPI endpoint, or a batch job with no changes.

---

## Key Engineering Decisions

**Why `@st.cache_resource` and not `@st.cache_data`?**
The model, scaler, and encoder are large objects that should be shared across users and reruns. `cache_resource` is the correct primitive for this — it holds a single instance per process. `cache_data` serializes and deserializes the return value on every call, which defeats the purpose for stateful ML objects.

**Why separate `validators.py` from `predictor.py`?**
Validation is testable without any ML artifacts. You can run the entire `test_validators.py` suite on a CI machine that has never seen the training notebook. Keeping validation logic inside the service class would force every test to mock artifact loading.

**Why pickle for the model artifact instead of `.h5`?**
The original repository shipped a `battery_life_model.pkl`. This is kept as the default to preserve drop-in compatibility. If you retrain and want to use `.h5`, update `MODEL_PATH` in `.env` — the predictor's `_load_artifact` method is format-agnostic for sklearn objects; for Keras `.h5` you'd swap `pickle.load` for `tf.keras.models.load_model`, which is a one-line change in `predictor.py`.

**Why mean imputation and not something more sophisticated?**
Imputing `Capacity` with the cross-cycle mean (which mixes discharge and non-discharge rows) introduces synthetic signal. A stricter approach would impute per cycle type or drop rows with missing target features entirely. Mean imputation is kept here because it matches the original notebook and the difference in a 7,565-row dataset with a simple ANN is small. The known-limitations section documents this explicitly.

---

## Local Setup

### Prerequisites

- Python 3.9 or newer
- `pip` 21+
- Git

### Install

```bash
git clone https://github.com/wittyswayam/EV_Battery_Life_Prediction_Using_ANN_Keras.git
cd EV_Battery_Life_Prediction_Using_ANN_Keras

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt
```

### Train the Model

```bash
jupyter notebook
# Open: notebooks/EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb
# Kernel → Restart & Run All
# Training takes 2–8 minutes on CPU.
# Output artifacts land in artifacts/
```

Verify all artifacts are present before running the app:

```bash
ls -lh artifacts/
# battery_life_model.pkl  scaler.pkl  label_encoder.pkl
```

### Run the App

```bash
streamlit run main.py
# Open http://localhost:8501
```

### Environment Variables

The app works with zero configuration. If you need to override artifact paths or log level, copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

```env
LOG_LEVEL=DEBUG
MODEL_PATH=artifacts/battery_life_model.pkl
```

Environment variables are loaded by `app/config.py` via `python-dotenv`.

---

## Docker

Before building the image, ensure all three artifacts are present in `artifacts/` (run the notebook first).

```bash
docker build -t ev-battery-prediction:latest .
docker run -d --name ev-battery-app -p 8501:8501 ev-battery-prediction:latest
# http://localhost:8501
```

Or with Compose:

```bash
docker compose up --build
```

The Compose file mounts `./logs` into the container so application logs persist on the host.

---

## Running Tests

```bash
pytest tests/ -v
```

`test_validators.py` has no external dependencies and runs in any environment. `test_predictor.py` includes a `TestPredictionServiceWithRealArtifacts` class that is automatically skipped if `artifacts/` is incomplete — so CI passes without a training step.

For coverage:

```bash
pytest tests/ --cov=app --cov-report=term-missing
```

### Linting & Formatting

```bash
black app/ main.py          # Format
isort app/ main.py           # Sort imports
flake8 app/ main.py          # Lint
```

---

## Using the Inference Service Programmatically

The `PredictionService` class can be imported directly — no Streamlit required:

```python
from app.predictor import PredictionService

service = PredictionService()  # loads artifacts once

temperature = service.predict(
    cycle_type="discharge",
    capacity=1.6743,
    re=0.056058,
    rct=0.200970,
)
print(f"{temperature:.2f} °C")
```

The method raises `app.validators.ValidationError` for malformed inputs and `RuntimeError` for inference failures. Both are documented and catchable.

---

## Known Limitations

**Random train/test split.** The 80/20 split is random, not temporal. Consecutive cycles from the same battery cell can appear in both train and test sets, which is a form of data leakage. A proper split would hold out the last N cycles per cell for testing. For a baseline ANN on this dataset size the difference is small, but it should be addressed before using evaluation metrics to make any production reliability claims.

**Cross-cycle mean imputation.** `Capacity`, `Re`, and `Rct` are each only physically meaningful for one cycle type. Filling discharge-cycle `Re` values with an imputed mean derived partly from impedance-cycle measurements introduces cross-type signal. Per-type imputation or simply masking these values before feeding them into the feature vector would be cleaner.

**Low-cardinality regression target.** `ambient_temperature` takes only three values (4, 24, 44 °C). Framing this as regression works and the model learns it, but a three-class classifier would likely outperform the ANN on this specific target. The regression framing is preserved because it matches the original formulation and is more extensible if the dataset is later expanded with continuous temperature recordings.

**No input distribution warnings.** The app accepts any finite float for capacity, Re, and Rct. Values far outside the training distribution (e.g., capacity = 100 Ah) will produce a prediction without any warning. A proper implementation would compute Z-scores against training statistics and surface an out-of-distribution alert to the user.

---

## CI/CD

The `.github/workflows/ci.yml` pipeline runs on every push to `main` or `develop` and on all pull requests targeting `main`. It:

1. Installs dependencies from `requirements.txt`
2. Checks formatting with `black --check`
3. Checks import order with `isort --check-only`
4. Runs `flake8` for PEP 8 compliance
5. Executes the full test suite with `pytest`

No training step runs in CI. The `TestPredictionServiceWithRealArtifacts` class is skipped automatically when artifacts are absent.

---

## Future Work

**FastAPI inference endpoint.** The `PredictionService` is already decoupled from Streamlit and can be wrapped in a FastAPI route in an afternoon. Adding a `/api/v1/predict` endpoint would make the model callable from external systems without opening the Streamlit UI.

**Per-cycle-type imputation.** Replace the global mean fill with per-type medians. This is a two-line change in the notebook but would eliminate the cross-type signal currently introduced by mean imputation.

**Temporal train/test split.** Split by cycle index rather than randomly to prevent future-data leakage within a single battery cell's sequence.

**State of Health target.** Derive SOH = `Capacity / NominalCapacity` per `battery_id` and predict it directly. This is a more operationally useful target than ambient temperature for fleet management applications.

**LSTM for sequential modelling.** The discharge cycles within a single battery cell form a sequence. An LSTM operating over the cycle history of a cell would capture degradation trends that the current feedforward ANN misses entirely.

**Artifact checksums.** In a production deployment where artifacts are fetched from object storage, verifying SHA-256 checksums before loading prevents silent model substitution and catches corrupted downloads.

---

## Dataset

NASA Prognostics Center of Excellence Battery Dataset:

> B. Saha and K. Goebel (2007). "Battery Data Set", NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.  
> https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://github.com/wittyswayam">wittyswayam</a> &nbsp;·&nbsp; NASA PCoE Battery Data &nbsp;·&nbsp; TensorFlow / Keras &nbsp;·&nbsp; Streamlit
</p>
