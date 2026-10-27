# ChurnIQ — Premium Streamlit Customer Churn App

A polished Streamlit interface for entering a customer's telecom account details and viewing a churn prediction with a visual probability gauge.

## 1. Put the files together

Keep these files in the same folder:

- `app.py`
- `requirements.txt`
- `best_model.pkl`
- `encoder.pkl`
- `scaler.pkl`
- Optional: `WA_Fn-UseC_-Telco-Customer-Churn.csv` for Portfolio Insights charts

The three `.pkl` files must come from your own trained notebook. They are not included in this starter app.

## 2. Run on Windows

Open Command Prompt or the VS Code terminal in this folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the app. If it does not, copy the local URL printed in the terminal (usually `http://localhost:8501`).

## 3. Connect your notebook

At the end of your training notebook, save the exact trained artifacts:

```python
import pickle

with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
```

Use the actual variable names from your notebook. If your notebook used a different encoder/scaler strategy, make sure the app's preprocessing matches it. The app can detect missing files, but valid predictions require identical feature order and preprocessing.

## 4. Important compatibility note

The app is designed around the Telco Customer Churn dataset. If your notebook encoded columns individually, save a dictionary of encoders or update the prediction preprocessing in `app.py` to mirror your notebook exactly. For a robust production setup, save a single scikit-learn `Pipeline` that includes preprocessing and the classifier, then pass raw customer rows directly into that pipeline.

## Features

- Dark, premium dashboard interface
- Guided customer profile form
- Churn / stay prediction and probability gauge
- Low / medium / high risk band with suggested next action
- Optional portfolio insights from the original CSV
- Sidebar navigation and model-file status
