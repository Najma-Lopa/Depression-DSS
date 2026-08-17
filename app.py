import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import shap
from streamlit_shap import st_shap

# --- Page Configuration & Custom CSS ---
st.set_page_config(page_title="Depression Diagnostic Dashboard", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    /* Premium Header Styling */
    .header-container {
        background: linear-gradient(135deg, #F0F4FF 0%, #D9E2EC 100%);
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #E2E8F0;
    }
    .main-title { 
        font-size: 42px; 
        font-weight: 800; 
        color: #1E3A8A; 
        margin-bottom: 10px; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .sub-title { 
        font-size: 18px; 
        color: #4B5563; 
        margin-bottom: 0px; 
        font-weight: 500;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# --- Data Loading and Model Training ---
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("dataset_before encoding.xlsx", sheet_name='USDI (Original)')
    
    encoders = {}
    encoded_df = df.copy()
    
    for col in df.columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    X = encoded_df.drop('Class', axis=1)
    y = encoded_df['Class']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return df, X, y, model, encoders

df, X, y, model, encoders = load_and_prepare_data()
features = X.columns.tolist()
classes = encoders['Class'].classes_

# --- Header Section (Centered & Styled) ---
st.markdown("""
    <div class="header-container">
        <div class="main-title">🧠 Depression Diagnostic Dashboard</div>
        <div class="sub-title">Predict severity and analyze impactful features using Machine Learning & XAI</div>
    </div>
""", unsafe_allow_html=True)

# --- Layout: 2 Columns ---
col_input, col_results = st.columns([1, 2], gap="large")

with col_input:
    st.subheader("📋 Patient Assessment Form")
    st.write("Please provide the responses to the following primary metrics:")
    
    with st.form("prediction_form"):
        user_input = {}
        important_cols = [
            'futureApathy', 'persistentSadness', 'selfWorth', 
            'failureFeeling', 'irritability', 'burdenFeelings', 
            'alienation', 'lossOfControl'
        ]
        
        form_col1, form_col2 = st.columns(2)
        
        for i, col in enumerate(important_cols):
            if i % 2 == 0:
                with form_col1:
                    user_input[col] = st.selectbox(f"{col}", df[col].unique())
            else:
                with form_col2:
                    user_input[col] = st.selectbox(f"{col}", df[col].unique())
            
        with st.expander("View all other clinical features"):
            for col in features:
                if col not in important_cols:
                    user_input[col] = st.selectbox(f"{col}", df[col].unique())
                    
        submit_button = st.form_submit_button(label="Predict Severity", use_container_width=True)

with col_results:
    if submit_button:
        st.subheader("📊 Diagnostic Results")
        
        # 1. Encode user input
        input_data = []
        for col in features:
            val = user_input[col]
            encoded_val = encoders[col].transform([val])[0]
            input_data.append(encoded_val)
            
        input_df = pd.DataFrame([input_data], columns=features)
        
        # 2. Prediction & Probabilities
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = encoders['Class'].inverse_transform([prediction_encoded])[0]
        probabilities = model.predict_proba(input_df)[0]
        
        # --- Dynamic UI Box & Suggestions based on Prediction ---
        if prediction_label == "No Depression":
            box_bg, box_border, text_color, icon = "#Edf7ed", "#4caf50", "#2e7d32", "✅"
            rec_bg, rec_border, rec_title = "#F4FBF4", "#4caf50", "#1B5E20"
            suggestion = (
                "<ul style='margin: 0; padding-left: 20px; color: #444; font-size: 15px; line-height: 1.8; font-family: sans-serif;'>"
                "<li><strong>Keep up the great work!</strong></li>"
                "<li>Maintain your healthy lifestyle and balanced routine.</li>"
                "<li>Ensure you continue getting enough sleep.</li>"
                "<li>Stay connected with friends and family.</li>"
                "</ul>"
            )
        
        elif prediction_label == "Mild Depression":
            box_bg, box_border, text_color, icon = "#fff4e5", "#ff9800", "#ed6c02", "⚠️"
            rec_bg, rec_border, rec_title = "#FFF9F0", "#ff9800", "#E65100"
            suggestion = (
                "<ul style='margin: 0; padding-left: 20px; color: #444; font-size: 15px; line-height: 1.8; font-family: sans-serif;'>"
                "<li>It seems you might be experiencing some <strong>mild stress</strong>.</li>"
                "<li>Try to incorporate relaxation techniques into your routine.</li>"
                "<li>Take regular, short breaks from your studies.</li>"
                "<li>Consider talking to someone you trust about how you feel.</li>"
                "</ul>"
            )
        
        elif prediction_label == "Moderate Depression":
            box_bg, box_border, text_color, icon = "#fdecea", "#ef5350", "#d32f2f", "🔔"
            rec_bg, rec_border, rec_title = "#FFF3F2", "#ef5350", "#B71C1C"
            suggestion = (
                "<ul style='margin: 0; padding-left: 20px; color: #444; font-size: 15px; line-height: 1.8; font-family: sans-serif;'>"
                "<li>Your metrics indicate <strong>moderate depressive symptoms</strong>.</li>"
                "<li>It is highly advisable to consult with a university counselor.</li>"
                "<li>Reach out to a healthcare professional to discuss your feelings.</li>"
                "<li>Explore proper coping strategies and do not hesitate to ask for help.</li>"
                "</ul>"
            )
        
        else: # Severe Depression
            box_bg, box_border, text_color, icon = "#ffebee", "#d32f2f", "#c62828", "🚨"
            rec_bg, rec_border, rec_title = "#FFF0F0", "#d32f2f", "#880E4F"
            suggestion = (
                "<ul style='margin: 0; padding-left: 20px; color: #444; font-size: 15px; line-height: 1.8; font-family: sans-serif;'>"
                "<li>The assessment indicates <strong>severe symptoms</strong>.</li>"
                "<li>Please prioritize your mental health immediately.</li>"
                "<li>Reach out to a medical professional, therapist, or counselor right away.</li>"
                "<li>Contact a mental health helpline for immediate proper support and guidance.</li>"
                "</ul>"
            )

        # Display Custom Result Box
        st.markdown(f"""
        <div style="background-color: {box_bg}; border: 1px solid {box_border}; border-radius: 12px; padding: 30px; text-align: center; margin-bottom: 20px;">
            <h1 style="color: {text_color}; margin: 0; font-size: 32px; display: flex; align-items: center; justify-content: center; gap: 10px;">
                {icon} {prediction_label}
            </h1>
            <p style="color: #555; margin-top: 10px; font-size: 16px;">Predicted based on your clinical and lifestyle metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Custom Recommendation Box
        st.markdown(f"""
        <div style="background-color: {rec_bg}; border-left: 8px solid {rec_border}; border-radius: 8px; padding: 20px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <h4 style="color: {rec_title}; margin-top: 0; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; font-family: sans-serif;">
                💡 Actionable Recommendations
            </h4>
            {suggestion}
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # --- Probability Bar Chart ---
        st.write("#### Probability Distribution")
        prob_df = pd.DataFrame({'Condition': classes, 'Probability': probabilities})
        prob_df = prob_df.sort_values(by='Probability', ascending=False) 
        
        # একদম ক্লিয়ার এবং ডিস্টিংক্ট কালার ম্যাপ 
        color_discrete_map = {
            "No Depression": "#28A745",       # Green
            "Mild Depression": "#FFC107",     # Yellow
            "Moderate Depression": "#FD7E14", # Orange
            "Severe Depression": "#DC3545"    # Red
        }

        fig_bar = px.bar(
            prob_df, 
            x='Condition', 
            y='Probability', 
            text='Probability',
            color='Condition', 
            color_discrete_map=color_discrete_map
        )
        
        fig_bar.update_traces(
            texttemplate='%{text:.1%}', 
            textposition='outside',
            textfont=dict(size=15, color='black', weight='bold') 
        )
        
        fig_bar.update_layout(
            height=380, 
            margin=dict(l=0, r=0, t=30, b=0), 
            showlegend=False, 
            yaxis=dict(
                range=[0, max(probabilities) * 1.2],
                title=dict(text="Probability", font=dict(size=16, color='black', weight='bold')),
                tickfont=dict(size=14, color='black')
            ),
            xaxis=dict(
                title=dict(text="Condition", font=dict(size=16, color='black', weight='bold')),
                tickfont=dict(size=14, color='black')
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="probability_bar_chart")

        st.divider()
        
        # --- SHAP Force Plot (Explainable AI) ---
        st.write("#### 🧠 Explainable AI (SHAP Force Plot)")
        st.write(f"Why did the model predict **{prediction_label}**? This interactive plot shows how specific features pushed the model's decision.")
        
        with st.spinner("Generating SHAP Explanation..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_df)
            
            if isinstance(shap_values, list):
                shap_val_for_class = shap_values[prediction_encoded][0]
                expected_val_for_class = explainer.expected_value[prediction_encoded]
            else:
                shap_val_for_class = shap_values[0, :, prediction_encoded]
                expected_val_for_class = explainer.expected_value[prediction_encoded]
            
            display_input_df = pd.DataFrame([user_input])[features]
            
            shap_fig = shap.force_plot(
                expected_val_for_class, 
                shap_val_for_class, 
                display_input_df.iloc[0], 
                feature_names=features,
                matplotlib=False
            )
            st_shap(shap_fig, height=150)
            
        st.divider()
        
        # --- Radar Chart (Feature Impact) ---
        st.write("#### Global Feature Impact Analysis")
        st.write("Which specific features generally contribute the most to this prediction?")
        
        importances = model.feature_importances_
        impact_scores = input_data * importances
        
        custom_weights = []
        for col in features:
            if col in important_cols:
                custom_weights.append(1.5) 
            else:
                custom_weights.append(1.0)
                
        adjusted_impact_scores = impact_scores * np.array(custom_weights)
        
        impact_df = pd.DataFrame({'Feature': features, 'Impact': adjusted_impact_scores})
        top_impact_df = impact_df.sort_values(by='Impact', ascending=False).head(6)
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=top_impact_df['Impact'].tolist() + [top_impact_df['Impact'].iloc[0]], 
            theta=top_impact_df['Feature'].tolist() + [top_impact_df['Feature'].iloc[0]],
            fill='toself',
            name='Impact Score',
            line_color='#3B82F6',
            fillcolor='rgba(59, 130, 246, 0.4)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=False)
            ),
            showlegend=False,
            height=350,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="feature_impact_radar_chart")
        
    else:
        st.info("👈 Please fill out the assessment form on the left and click **Predict Severity** to see the results, actionable recommendations, and feature impact analysis.")