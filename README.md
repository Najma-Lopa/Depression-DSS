# 🧠 Depression Diagnostic & XAI Dashboard

A state-of-the-art web application built with **Streamlit** and **Machine Learning** designed to predict university students' depression severity based on clinical and lifestyle metrics. It integrates **Explainable AI (XAI)** to provide transparent, interpretable diagnostics and actionable health recommendations.

🔗 **Live App:** [https://depression-dss.streamlit.app/](https://depression-dss.streamlit.app/)

---

## ✨ Key Features

* **Interactive Patient Assessment:** 
  * Easy-to-use input form optimized for primary mental health indicators (e.g., *futureApathy*, *persistentSadness*, *selfWorth*, *failureFeeling*, *irritability*, *burdenFeelings*, *alienation*, and *lossOfControl*).
  * Advanced clinical features neatly organized inside a collapsible expander.
* **Real-Time Machine Learning Prediction:** 
  * Powered by robust ensemble classification models (Random Forest) trained on comprehensive student mental health datasets.
* **Dynamic Diagnostic & Severity Box:** 
  * Visually highlights the predicted depression condition (*No Depression*, *Mild Depression*, *Moderate Depression*, or *Severe Depression*) with color-coded alerts and icons.
* **Actionable Recommendations:** 
  * Dynamically generated professional guidance and bullet-pointed coping strategies tailored to the specific severity level.
* **Probability Distribution Chart:** 
  * Clear vertical probability distribution bar chart showing the percentage likelihood across all severity classes with distinct semantic color-coding.
* **Explainable AI (XAI) - SHAP Integration:** 
  * Built-in SHAP feature contribution plots to explain *why* the model made a specific prediction, ensuring medical and academic transparency.
* **Global Feature Impact Analysis:** 
  * Interactive radar chart displaying the most influential psychological and lifestyle factors contributing to the assessment.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit (Python Web Framework)
* **Machine Learning:** Scikit-Learn (Random Forest Classifier, Label Encoding)
* **Explainable AI:** SHAP (SHapley Additive exPlanations)
* **Data Visualization:** Plotly, Matplotlib, Seaborn
* **Data Processing:** Pandas, NumPy, Openpyxl

---

## 🚀 Getting Started Locally

Follow these instructions to run the application on your local machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/depression-dss.git](https://github.com/your-username/depression-dss.git)
cd depression-dss

```

### 2. Install Dependencies

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt

```

### 3. Run the Streamlit App

```bash
streamlit run app.py

```

---

## 📦 `requirements.txt`

Ensure your repository includes a `requirements.txt` file with the following dependencies for seamless deployment on Streamlit Cloud:

```text
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
shap
matplotlib
streamlit-shap
xgboost
catboost
seaborn

```

---

## 💡 Usage Guide

1. Open the live web app: [https://depression-dss.streamlit.app/](https://depression-dss.streamlit.app/)
2. Fill out the primary psychological metrics in the **Patient Assessment Form**.
3. Expand the clinical features section if you want to fine-tune other parameters.
4. Click **Predict Severity** to instantly view the diagnostic breakdown, probability percentages, XAI explanations, and professional recommendations.
