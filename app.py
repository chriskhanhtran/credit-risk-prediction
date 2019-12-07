# https://streamlit.io/docs/api.html
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib

st.markdown('# FICO Default Risk Prediction')

# Load model
model_names = ['rfc', 'svc', 'lgbm']

@st.cache(show_spinner=False)
def load_model(model_names):
    models = []
    for model_name in ['rfc', 'svc', 'lgbm']:
        filename = 'models/' + model_name + '.joblib.pkl'
        model = joblib.load(filename)
        models.append(model)
    return models

models = load_model(model_names)
rfc, svc, lgbm = models

# Select model
selected_model = st.selectbox('Model: ',
                              ['Random Forest', 'Support Vectors Machine',
                               'LightGBM'])

if selected_model == 'Random Forest':
    model = rfc
elif selected_model == 'Support Vectors Machine':
    model = svc
elif selected_model == 'LightGBM':
    model = lgbm

# Fill values
st.sidebar.markdown("**Variables:**")
X = {}

X['ExternalRiskEstimate'] = st.sidebar.slider('ExternalRiskEstimate', -9, 94, 67)
X['MSinceOldestTradeOpen'] = st.sidebar.slider('MSinceOldestTradeOpen', -9, 803, 184)
X['MSinceMostRecentTradeOpen'] = st.sidebar.slider('MSinceMostRecentTradeOpen', -9, 383, 8)
X['AverageMInFile'] = st.sidebar.slider('AverageMInFile', -9, 383, 73)
X['NumSatisfactoryTrades'] = st.sidebar.slider('NumSatisfactoryTrades', -9, 79, 19)
X['NumTrades60Ever2DerogPubRec'] = st.sidebar.slider('NumTrades60Ever2DerogPubRec', -9, 19, 0)
X['NumTrades90Ever2DerogPubRec'] = st.sidebar.slider('NumTrades90Ever2DerogPubRec', -9, 19, 0)
X['PercentTradesNeverDelq'] = st.sidebar.slider('PercentTradesNeverDelq', -9, 100, 86)
X['MSinceMostRecentDelq'] = st.sidebar.slider('MSinceMostRecentDelq', -9, 83, 6)
X['MaxDelq2PublicRecLast12M'] = st.sidebar.slider('MaxDelq2PublicRecLast12M', -9, 9, 5)
X['MaxDelqEver'] = st.sidebar.slider('MaxDelqEver', -9, 8, 5)
X['NumTotalTrades'] = st.sidebar.slider('NumTotalTrades', -9, 104, 20)
X['NumTradesOpeninLast12M'] = st.sidebar.slider('NumTradesOpeninLast12M', -9, 19, 1)
X['PercentInstallTrades'] = st.sidebar.slider('PercentInstallTrades', -9, 100, 32)
X['MSinceMostRecentInqexcl7days'] = st.sidebar.slider('MSinceMostRecentInqexcl7days', -9, 24, 0)
X['NumInqLast6M'] = st.sidebar.slider('NumInqLast6M', -9, 66, 0)
X['NumInqLast6Mexcl7days'] = st.sidebar.slider('NumInqLast6Mexcl7days', -9, 66, 0)
X['NetFractionRevolvingBurden'] = st.sidebar.slider('NetFractionRevolvingBurden', -9, 232, 32)
X['NetFractionInstallBurden'] = st.sidebar.slider('NetFractionInstallBurden', -9, 471, 39)
X['NumRevolvingTradesWBalance'] = st.sidebar.slider('NumRevolvingTradesWBalance', -9, 32, 3)
X['NumInstallTradesWBalance'] = st.sidebar.slider('NumInstallTradesWBalance', -9, 23, 1)
X['NumBank2NatlTradesWHighUtilization'] = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', -9, 18, 0)
X['PercentTradesWBalance'] = st.sidebar.slider('PercentTradesWBalance', -9, 100, 62)

# Convert X to DataFrame
X = pd.DataFrame(X, index=[0])


# Show prediction
def show_prediction():
    #if st.button("Run Model"):
    pred = model.predict_proba(X)[0, 1]
    st.markdown(f'## Risk of Default: {pred*100:.2f} %')
        
show_prediction()


# Show model's evaluation
def show_evaluation():
    if st.checkbox("Show Model's Evaluation"):
        if selected_model == 'Random Forest':
            st.write('**AUC:** 0.79')
            st.write('**Accuracy:** 72.47%')
            image = Image.open('img/rfc.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'Support Vectors Machine':
            st.write('**AUC:** 0.78')
            st.write('**Accuracy:** 72.75%')
            image = Image.open('img/svc.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'LightGBM':
            st.write('**AUC:** 0.80')
            st.write('**Accuracy:** 73.14%')
            image = Image.open('img/lgbm.png')
            st.image(image, use_column_width=False)
            
show_evaluation()

