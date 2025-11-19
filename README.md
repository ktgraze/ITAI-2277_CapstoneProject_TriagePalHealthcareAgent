# 🩺 TriagePal: AI Multi-Modal Healthcare Triage Agent  

[![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)](https://itai-2277capstoneprojecttriagepalhealthcareagent-bl9hwtxu9uskl.streamlit.app/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-CNN%20%2B%20BioBERT%20%2B%20RandomForest-blue)]()

---

## ✨ Live Demo

The TriagePal agent is deployed on **Streamlit Community Cloud** and ready for real-time interaction.

👉 **Try the live app here:**  
### https://itai-2277capstoneprojecttriagepalhealthcareagent-bl9hwtxu9uskl.streamlit.app/

## Screenshots

### Home Screen - Upload Interface
![Home Screen](images/home_screen.png)

### Analysis Results - Summary & Report
![Results Screen](images/results_screen.png)

### Recent Analyses Table
![Recent Analyses](images/recent_analyses.png)

---

## 🎯 Project Overview (ITAI-2277 Capstone)

TriagePal is the **Phase 4 Integrated System Prototype** for the *AI Applications and Resources (ITAI-2277)* course at **Houston Community College**.

It is designed to provide a **preliminary, AI-supported triage score** using both:

- 🖼️ **Image analysis** via a CNN  
- 📝 **Symptom text analysis** using BioBERT NLP  
- 🌲 **Random Forest meta-classifier** to integrate both modalities  

### 🧠 How It Works
The system uses a multi-step intelligence pipeline:

1. **Computer Vision (CV)**  
   A TensorFlow CNN model analyzes an uploaded eye image to detect signs of infection.

2. **Natural Language Processing (NLP)**  
   A BioBERT model extracts symptoms and severity indicators from the user’s text.

3. **Triage Integration**  
   A Random Forest classifier combines:
   - CNN prediction  
   - Symptom count  
   - NLP-derived urgency score  
   
   → Producing a **Low, Medium, or High urgency** triage label.

---

## ⚙️ Technology Stack

| Component | Technology / Model | Purpose |
|----------|--------------------|---------|
| **User Interface** | Streamlit | Provides the interactive web app |
| **Image Model** | TensorFlow CNN (`triagepal_optimized_model.h5`) | Predicts condition from image |
| **NLP Model** | BioBERT (`dmis-lab/biobert-v1.1`) | Extracts symptoms and urgency |
| **Triage Logic** | Scikit-learn Random Forest (`rf_triage_agent.pkl`) | Final urgency classification |
| **Database** | SQLite (`triagepal.db`) | Stores session reports |
| **Deployment** | Streamlit Cloud + GitHub | Hosts the live application |

### **Best Model Performance (Phase 3 Evaluation)**  
- **Test Accuracy:** 0.9307  
- **ROC–AUC:** 0.9870  

---

## 📁 Repository Structure

