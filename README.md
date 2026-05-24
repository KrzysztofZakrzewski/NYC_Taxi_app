# 🚕 NYC Taxi Trip Duration Predictor — Streamlit App

## 📌 Overview

This application provides an interactive machine learning playground for predicting NYC taxi trip duration using a trained Random Forest regression model.

Users can manually enter custom trip information such as:

* 🗺️ pickup and dropoff coordinates
* 👥 passenger count
* ⏰ pickup time information
* 📅 weekday and monthly data
* 🚕 vendor information

The application automatically performs feature engineering in real time, including:

* 📏 Haversine trip distance calculation
* 🧭 directional movement (`bearing`)
* 🔄 cyclic temporal encoding (`hour_sin`, `hour_cos`)
* 📆 weekend detection

The final prediction is generated using the trained Random Forest model developed during the NYC Taxi Trip Duration machine learning project.

---

# 🤖 Model Information

## Final Model

* Random Forest Regressor

## Model Performance

* MAE ≈ 184 seconds
* RMSE ≈ 273 seconds
* R² ≈ 0.80

The model was trained on a cleaned and optimized 200k subset derived from the original 1.4 million NYC taxi ride dataset.

---

# 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib

---

# 🚀 Installation

## 1. Clone Repository

```bash
git clone YOUR_REPOSITORY_LINK
cd YOUR_PROJECT_FOLDER
```

## 2. Create Conda Environment

```bash
conda env create -f environment.yml
```

## 3. Activate Environment

```bash
conda activate taxi_app
```

## 4. Run Application

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```bash
project/
│
├── app.py
├── environment.yml
├── models/
│   └── random_forest_model_sample.pkl
└── README.md
```

---

# 🎮 Usage

After launching the application:

1. Enter custom trip information
2. Click:

```text
🚀 Predict Trip Duration
```

3. The model will automatically:

* calculate trip distance
* generate directional movement features
* apply temporal encoding
* predict taxi trip duration

The final prediction is displayed in:

* seconds
* minutes

along with selected auto-generated transportation features.

---

# 👨‍💻 Author

Krzysztof Zakrzewski
