from flask import Flask, render_template, request, redirect, url_for, send_file, flash
import os
import csv
import pandas as pd
from utils.predictor import load_model, predict_applicant

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET', 'smart-lender-secret')

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)
PREDICTIONS_CSV = os.path.join('logs', 'predictions.csv')

# Load model and preprocessing artifacts
MODEL, ARTIFACTS = load_model('model.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form.to_dict()
        result = predict_applicant(MODEL, ARTIFACTS, data)
        # log prediction
        os.makedirs(os.path.dirname(PREDICTIONS_CSV), exist_ok=True)
        header = ['timestamp'] + list(result['input'].keys()) + ['predicted', 'probability']
        write_header = not os.path.exists(PREDICTIONS_CSV)
        with open(PREDICTIONS_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([pd.Timestamp.now()] + list(result['input'].values()) + [result['predicted'], result['probability']])

        return render_template('result.html', result=result)

    return render_template('predict.html')


@app.route('/dashboard')
def dashboard():
    df = None
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
    return render_template('dashboard.html', df=df)


@app.route('/history')
def history():
    df = None
    if os.path.exists(PREDICTIONS_CSV):
        df = pd.read_csv(PREDICTIONS_CSV)
    return render_template('history.html', df=df)


@app.route('/download')
def download():
    if os.path.exists(PREDICTIONS_CSV):
        return send_file(PREDICTIONS_CSV, as_attachment=True)
    flash('No predictions available to download yet.', 'warning')
    return redirect(url_for('dashboard'))


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
