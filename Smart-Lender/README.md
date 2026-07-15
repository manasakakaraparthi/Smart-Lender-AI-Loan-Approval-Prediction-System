# Smart Lender

AI-powered loan approval demo application using Flask and scikit-learn/XGBoost.

Setup

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py  # prepare models using dataset/loan_data.csv
python app.py
```

Notes
- Add a real dataset at `dataset/loan_data.csv` with a `Loan_Status` or `Target` column.
- The `train_model.py` script will save the best model to `models/best_model.pkl` and a `models/scaler.pkl`.
