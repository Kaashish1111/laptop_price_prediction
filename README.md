# laptop price prediction
AI laptop price prediction 


An end-to-end Machine Learning web application that predicts laptop prices based on their specifications. The project includes data preprocessing, feature engineering, model comparison, hyperparameter tuning, and deployment using Streamlit.

## Live Demo

**Live App:** https://laptoppricepredection.streamlit.app/

---

## Features

- Predicts laptop prices based on hardware specifications
- Clean and interactive Streamlit interface
- Advanced feature engineering
- ElasticNetCV model with hyperparameter tuning
- Real-time price prediction

---

##  Dataset

- **Total Samples:** 893 Laptops
- **Target Variable:** Price
- **Task:** Regression

---

## Feature Engineering

The following preprocessing and feature engineering steps were performed:

- Removed unnecessary columns
- Converted RAM and ROM into numerical values
- Extracted:
  - Processor Brand
  - Processor Family
  - Processor Generation
- Extracted:
  - GPU Brand
  - VRAM
  - Dedicated GPU
- Extracted:
  - CPU Core Count
  - Thread Count
- Created Laptop Categories
- Created Screen Resolution (Pixels)
- One-Hot Encoding for categorical variables
- Feature Scaling using StandardScaler

---

## Machine Learning Models

The following regression models were trained and evaluated:

- Linear Regression
- Ridge Regression (RidgeCV)
- Lasso Regression (LassoCV)
- ElasticNet Regression (ElasticNetCV)

### Best Model

**ElasticNetCV**

---

## Evaluation Metrics

The models were evaluated using:

- R² Score
- Adjusted R²
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib

---

## Project Structure

```
Laptop_Price_Prediction/
│
├── app.py
├── elastic_model.pkl
├── scaler.pkl
├── model_columns.pkl
├── requirements.txt
├── README.md
├── laptop_price_prediction.ipynb
└── data.csv
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/kaashish1111/Laptop_Price_Prediction.git
```

Move into the project folder

```bash
cd Laptop_Price_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---


##  Author

**Kashish **

GitHub: https://github.com/Kaashish1111

LinkedIn: *https://www.linkedin.com/in/kashishgoyal111/*

---

## ⭐ If you like this project

Please consider giving the repository a ⭐ on GitHub.
