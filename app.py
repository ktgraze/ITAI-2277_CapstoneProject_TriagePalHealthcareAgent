import streamlit as st
from PIL import Image
import sqlite3
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
import datetime

# Database connection
conn = sqlite3.connect('triagepal.db')
cursor = conn.cursor()

# Load optimized models
model = tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/triagepal_optimized_model.h5')
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()  # Load trained RF (placeholder; load actual model)

# Preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# NLP with BioBERT
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model_nlp = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")

def parse_symptoms_advanced(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    outputs = model_nlp(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    symptoms = [tokens[i] for i, pred in enumerate(predictions[0]) if pred != 0]
    urgency_score = sum(1 for sym in symptoms if sym.lower() in ['redness', 'swelling', 'pain'])
    return {"symptoms": symptoms, "urgency": min(urgency_score, 2)}

# Process input
def process_input(img_path, symptoms_text, save_to_db=True):
    cv_pred = model.predict(preprocess_image(Image.open(img_path)))
    nlp_result = parse_symptoms_advanced(symptoms_text)
    features = np.concatenate([cv_pred[0], [nlp_result["urgency"], len(nlp_result["symptoms"])])
    triage_score = rf.predict([features])[0]  # Use trained RF
    urgency = ["low", "medium", "high"][triage_score]

    patient_summary = f"This condition appears to be {urgency} urgency based on image and {len(nlp_result['symptoms'])} symptoms. " + \
                     ("Consult a doctor soon." if urgency in ["medium", "high"] else "Monitor and seek advice if worsening.")
    clinician_report = f"Triage score: {urgency} (CV: {cv_pred[0].argmax()}, Symptoms urgency: {nlp_result['urgency']}, Symptom count: {len(nlp_result['symptoms'])}). " + \
                      ("Urgent review recommended." if urgency == "high" else "Follow-up suggested.")

    if save_to_db:
        cursor.execute('''
        INSERT INTO triage_records (image_path, symptoms, cv_class, confidence, triage_score, patient_summary, clinician_report)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (img_path, symptoms_text, urgency, float(cv_pred[0].max()), urgency, patient_summary, clinician_report))
        conn.commit()

    return patient_summary, clinician_report

# Streamlit UI
st.title("TriagePal: AI Pre-Analysis Tool")
st.write("Upload an image and describe symptoms to get a preliminary assessment.")

uploaded_image = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "png"])
symptoms = st.text_area("Describe symptoms and conditions (e.g., redness, itch, migraines)")

if st.button("Analyze"):
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", width=300)

        # Save uploaded image temporarily
        img_path = f"/content/{uploaded_image.name}"
        with open(img_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Analyze
        try:
            patient_summary, clinician_report = process_input(img_path, symptoms)
            st.subheader("Patient Summary")
            st.write(patient_summary)
            st.subheader("Clinician Report")
            st.write(clinician_report)
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please upload an image to proceed.")

# Display recent records
st.subheader("Recent Analyses")
cursor.execute('SELECT * FROM triage_records ORDER BY timestamp DESC LIMIT 5')
records = cursor.fetchall()
for record in records:
    st.write(f"**Time:** {record[3]}, **Class:** {record[4]}, **Triage:** {record[6]}")

st.warning("This is not a medical diagnosis. Consult a healthcare professional for accurate advice.")
conn.close()  # Close on app exit (may not persist in Colab)
