from flask import Flask, render_template, request, redirect
import pandas as pd
import joblib
import os
from sentence_transformers import SentenceTransformer

app = Flask(__name__)


svm_classifier = joblib.load("model.pkl")
sbert_model = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

data_file = "complaints/data.csv"


if not os.path.exists(data_file):
    df = pd.DataFrame(columns=["Name", "Order ID", "Complaint", "Category"])
    df.to_csv(data_file, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def submit_complaint():
    name = request.form['name']
    order_id = request.form['order_id']
    complaint = request.form['complaint']
    
    
    complaint_embedding = sbert_model.encode([complaint], convert_to_tensor=False)

    
    category_index = svm_classifier.predict(complaint_embedding)[0]
    category = label_encoder.inverse_transform([category_index])[0]

    
    df = pd.read_csv(data_file)
    new_entry = pd.DataFrame([[name, order_id, complaint, category]], columns=df.columns)
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(data_file, index=False)

    return redirect(f'/success?name={name}&order_id={order_id}')

@app.route('/success')
def success():
    name = request.args.get('name')
    order_id = request.args.get('order_id')
    return render_template('success.html', name=name, order_id=order_id)

if __name__ == '__main__':
    app.run(debug=True)
