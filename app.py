import streamlit as st
from PIL import Image
import sqlite3
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
from pathlib import Path
import datetime
import pickle
import os

# Database connection
conn = sqlite3.connect('triagepal.db')
cursor = conn.cursor()

# Define the model architecture as it was when `triagepal_optimized_model.h5` was saved
# This function rebuilds the model, allowing us to load weights instead of the full model config.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense, BatchNormalization

def build_model_for_loading(conv1_units, conv2_units, conv3_units, dropout_rate, num_classes):
    model = Sequential()
    model.add(Conv2D(conv1_units, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(conv2_units, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(conv3_units, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Assumed (or known) hyperparameters and number of classes for the conjunctivitis model
# These values are based on the structure defined in cell lcw7SmvRmz5h and the binary classification nature.
NUM_CLASSES_CONJUNCTIVITIS = 2
OPT_CONV1_UNITS = 32 # min_value from tuner search
OPT_CONV2_UNITS = 256 # Corrected based on shape mismatch error
OPT_CONV3_UNITS = 384 # Corrected based on latest shape mismatch error
OPT_DROPOUT_RATE = 0.3 # A common tuned value, or assume default from range

# Build the model architecture
model = build_model_for_loading(OPT_CONV1_UNITS, OPT_CONV2_UNITS, OPT_CONV3_UNITS, OPT_DROPOUT_RATE, NUM_CLASSES_CONJUNCTIVITIS)
# Load the saved weights into the reconstructed model
model.load_weights('models/triagepal_optimized_model(3).h5')

# Load trained RandomForestClassifier
try:
    with open('models/rf_triage_agent.pkl', 'rb') as f:
        rf = pickle.load(f)
except FileNotFoundError:
    st.error("Error: Trained Random Forest model not found. Please ensure 'rf_triage_agent.pkl' is in /content/.")
    st.stop() # Stop the app if the model isn't found

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
    symptoms = [tokens[i] for i, pred in enumerate(predictions[0]) if pred != 0] # Non-0 labels are entities
    urgency_score = sum(1 for sym in symptoms if sym.lower() in ['redness', 'swelling', 'pain'])
    return {"symptoms": symptoms, "urgency": min(urgency_score, 2)}

# Process input
def process_input(img_path, symptoms_text, save_to_db=True):
    cv_pred = model.predict(preprocess_image(Image.open(img_path)))
    nlp_result = parse_symptoms_advanced(symptoms_text)
    # Ensure features array has correct dimensions for rf.predict
    features = np.concatenate([cv_pred[0], [nlp_result["urgency"], len(nlp_result["symptoms"])]])
    features = features.reshape(1, -1) # Reshape for single sample prediction

    triage_score = rf.predict(features)[0]
    urgency = ["low", "medium", "high"][int(triage_score)] # Ensure triage_score is integer for indexing

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
        img_path = uploaded_image.name
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
# Removed conn.close() from here, as it might prematurely close the connection
# for subsequent operations. It's better to manage connection lifecycle in a web app context.
