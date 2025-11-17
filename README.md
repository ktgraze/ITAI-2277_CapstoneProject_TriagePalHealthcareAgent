# 🩺 TriagePal: AI Multi-Modal Healthcare Triage Agent

[![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)]([DEPLOYMENT_URL_HERE])
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-MobileNetV2%20%2B%20RandomForest-blue)]()

## ✨ Live Demo

The TriagePal agent is hosted on Streamlit Community Cloud and is ready for use.

**[Click here to try the TriagePal Agent!]([PASTE YOUR STREAMLIT DEPLOYMENT URL HERE])**

---

## 🎯 Project Overview (ITAI-2277 Capstone)

The TriagePal agent is a **Phase 4 Integrated System Prototype** designed for our AI Applications and Resources course. Its primary goal is to provide preliminary, AI-driven triage for common eye conditions, specifically conjunctivitis, by leveraging **multi-modal input**—meaning it analyzes both images and text symptoms.

### How it Works

The agent processes user input in two main stages to generate a final, combined Triage Score (Low, Medium, or High Urgency):

1.  **Computer Vision (CV):** Analyzes an uploaded eye image to classify it as 'healthy' or 'infected'.
2.  **Natural Language Processing (NLP):** Parses user-provided text symptoms (e.g., "severe pain," "redness") to assign an initial urgency score.
3.  **Triage Integration:** A **Random Forest Classifier** acts as a meta-agent, combining the CV prediction, the symptom count, and the NLP urgency score to determine the final, comprehensive risk recommendation.

---

## ⚙️ Technology Stack

| Component | Technology / Model | Purpose |
| :--- | :--- | :--- |
| **User Interface (UI)** | **Streamlit** | Provides the simple, interactive web app for users to upload files and input symptoms. |
| **Image Model** | **TensorFlow / MobileNetV2** (`triagepal_optimized_model.h5`) | A CNN-based Transfer Learning model used for eye condition classification. |
| **Triage Logic** | **Scikit-learn / Random Forest Classifier** (`rf_triage_agent.pkl`) | The final model used to weigh and combine results from the image and text inputs. |
| **Deployment** | **GitHub & Streamlit Cloud** | Hosts the application and manages dependencies via `requirements.txt`. |

**Best Model Performance (from Phase 3 Evaluation):**
* **Test Accuracy:** 0.9307
* **ROC-AUC:** 0.9870

---

## 🚀 Setup for Local Development

To run the TriagePal agent locally, follow these steps:

### Prerequisites

You need Python 3.8+ installed on your system.

### 1. Clone the Repository

```bash
git clone [https://github.com/ITAI-2277/ITAI-2277_CapstoneProject_TriagePalHealthcareAgent.git](https://github.com/ITAI-2277/ITAI-2277_CapstoneProject_TriagePalHealthcareAgent.git)
cd ITAI-2277_CapstoneProject_TriagePalHealthcareAgent
```

---

### 🧑‍💻 **Team Information**

**Course:** ITAI-2277: AI Applications and Resources

**Team Members:** Jazmine Brown, Javon Darby, Jeffery Dirden, Katherine Stanton
