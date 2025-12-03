# **EV Battery Life Prediction Using ANN (Keras)**

*A Complete Machine Learning System for Predicting Electric Vehicle Battery Health*

---

## 📝 **Introduction**

Electric Vehicle (EV) batteries degrade over time due to chemical aging, temperature variations, resistance changes, and daily usage patterns. Predicting this degradation is essential for:

* Battery health monitoring
* Preventive maintenance
* Warranty claims
* Charging optimization
* Extending EV lifespan

This project uses a **deep learning–based Artificial Neural Network (ANN)** to predict the battery’s health/remaining life using measurable electrochemical features.

The project includes:

* A complete **machine learning pipeline**
* A **trained neural network model** saved for deployment
* A **Streamlit-based prediction app**
* A fully documented **training notebook**
* The dataset used for training

This README explains everything in detail: architecture, data, preprocessing, feature engineering, model design, and deployment.

---

# 📁 **Repository Structure**

```
EV_Battery_Life_Prediction_Using_ANN_Keras/
│
├── EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb   # Training & analysis notebook
├── app.py                                             # Streamlit prediction app
├── metadata.csv                                       # Full dataset
├── battery_life_model.pkl                             # Trained ANN model
├── scaler.pkl                                         # StandardScaler for inputs
├── label_encoder.pkl                                  # Categorical encoder
└── README.md                                          
```

### What each file represents:

| File                                               | Purpose                                                        |
| -------------------------------------------------- | -------------------------------------------------------------- |
| `metadata.csv`                                     | Dataset used for training ML model                             |
| `EV_Battery_Life_Prediction_Using_ANN_Keras.ipynb` | Data exploration, preprocessing, ANN training                  |
| `app.py`                                           | Web app for real-time predictions                              |
| `battery_life_model.pkl`                           | Trained neural network                                         |
| `scaler.pkl`                                       | Scaler applied during training (MUST be used during inference) |
| `label_encoder.pkl`                                | Encodes discharge type → numerical                             |
| `README.md`                                        | Documentation                                                  |

---

# 📊 **Dataset Description (metadata.csv)**

The dataset contains **7,565 rows × 10 columns** of real battery discharge/charge measurements.

### **Columns Overview**

| Column Name           | Description                                         | Type        |
| --------------------- | --------------------------------------------------- | ----------- |
| `type`                | Charge or discharge profile type                    | Categorical |
| `start_time`          | Start time of the test                              | Timestamp   |
| `ambient_temperature` | Surrounding temperature during the test             | Numeric     |
| `battery_id`          | Unique battery identifier                           | Categorical |
| `test_id`             | Measurement/Test number                             | Categorical |
| `uid`                 | Unique record ID                                    | Categorical |
| `filename`            | Original data source file                           | Text        |
| `Capacity`            | **Battery capacity measurement** (target candidate) | Numeric     |
| `Re`                  | Electrolyte/internal resistance                     | Numeric     |
| `Rct`                 | Charge-transfer resistance                          | Numeric     |

---

# 🔍 **Detailed Data Analysis**

### ✔ Missing Values

* **Capacity:** ~40% missing
* **Re & Rct:** ~75% missing
* These missing patterns indicate experimental limitations.

### ✔ Numeric Formatting Issues

Some numeric values were represented as **strings** → they must be converted to float before training.

### ✔ Potential Target Variable

You may choose:

#### **Option A:** Predict *Capacity*

→ Measures current battery performance.

#### **Option B:** Predict *State of Health (SOH)*

SOH can be derived:

```
SOH = Capacity / InitialCapacity(battery_id) * 100
```

#### **Option C:** Predict Remaining Useful Life (RUL)

Requires sequential time-series → LSTM possible in future.

In the current project, **Capacity** is assumed the target.

---

# 🧠 **Machine Learning Pipeline**

The training notebook follows a complete ML pipeline:

---

## **1. Data Cleaning**

Tasks performed:

* Removal of corrupted or invalid entries
* Conversion of numeric strings to `float`
* Handling missing values (drop or impute)
* Encoding categorical columns

Example code:

```python
df['Capacity'] = pd.to_numeric(df['Capacity'], errors='coerce')
df['Re'] = pd.to_numeric(df['Re'], errors='coerce')
df['Rct'] = pd.to_numeric(df['Rct'], errors='coerce')
df = df.dropna(subset=['Capacity'])
```

---

## **2. Feature Engineering**

Features chosen for ANN:

* Encoded discharging type (`type`)
* Capacity
* Internal resistance (`Re`)
* Charge-transfer resistance (`Rct`)
* Ambient temperature

Categorical encoding:

```python
label_encoder = LabelEncoder()
df['type_encoded'] = label_encoder.fit_transform(df['type'])
```

Scaling:

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## **3. Train–Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
```

---

# 🔬 **Neural Network Architecture**

Implemented using **Keras Sequential API**.

### **Layers**

* Dense(64, ReLU)
* Dropout(0.2)
* Dense(32, ReLU)
* Dropout(0.2)
* Dense(1, Linear)

---

### **Compilation**

```python
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae']
)
```

### **Training**

* **150 epochs**
* **Batch size: 32**
* **Validation split applied**

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,
    batch_size=32
)
```

---

# 📈 **Evaluation Metrics**

The ANN is evaluated using:

| Metric       | Meaning                           |
| ------------ | --------------------------------- |
| **MAE**      | Average absolute prediction error |
| **MSE**      | Penalizes large errors            |
| **RMSE**     | Root mean squared error           |
| **R² Score** | Goodness of fit                   |

Example:

```python
pred = model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", mean_squared_error(y_test, pred, squared=False))
print("R2:", r2_score(y_test, pred))
```

---

# 💾 **Model Saving**

The trained ANN is serialized using:

```python
model.save("battery_life_model.h5")
```

Additionally, supporting preprocessing objects such as `StandardScaler` and `LabelEncoder` are stored as `.pkl` files.

---

# 🌐 **Prediction App (Streamlit)**

`app.py` contains a full web-based prediction app.

### **What it does:**

* Loads trained ANN model
* Loads scaler & label encoder
* Accepts user inputs:

  * Discharge type
  * Capacity
  * Resistance values
* Preprocesses the inputs
* Returns the predicted battery life value

### **Run the App**

```bash
streamlit run app.py
```

Access at:

```
http://localhost:8501/
```

---

# ⚙️ **How to Run the Project Locally**

### **1. Clone the repository**

```bash
git clone https://github.com/wittyswayam/EV_Battery_Life_Prediction_Using_ANN_Keras.git
cd EV_Battery_Life_Prediction_Using_ANN_Keras
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install tensorflow pandas numpy scikit-learn streamlit matplotlib
```

### **3. Train the model (optional)**

Run the notebook.

### **4. Launch the prediction app**

```bash
streamlit run app.py
```

---

# 🚀 **Future Improvements**

Here are upgrades you can add later:

### 🔧 **Model Improvements**

* LSTM/GRU model on time-series discharge data
* XGBoost/LightGBM baseline models
* Hyperparameter tuning (KerasTuner)

### 🌡 **Better Feature Engineering**

* SOH calculation per battery
* Temperature cycle mapping
* Capacity fade curves

### 🎛 **Deployment**

* Host Streamlit app on Streamlit Cloud
* Deploy Flask API + React frontend
* Use Docker for containerization

### 🧠 **Explainability**

* Use SHAP to show how each feature impacts predictions
* Add importance plot inside app

---
