# https://streamlit.io/docs/api.html
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
#####################################################################################
def document_page():
    st.markdown("""
    ## Thank you so much for your time!
    A full report, notebook, and GitHub repository can be found below:
    - [Report]()
    - [Notebook]()
    - [GitHub](https://github.com/chriskhanhtran/fico-default-risk)
    
    
    ##About Us
    This website is an interactive interface for our Machine Learning models for credit risk prediction, as a part of our final project for the Advanced Predictive Analytics course at Simon Business School, University of Rochester.
    
    Our goal for this project is not only to find a relatively accurate and robust model for risk prediction, but more importantly, to improve its interpretability and give understandable explanations for sales representatives in a bank/credit card company can use to decide on accepting or rejecting applications.
    
    If you have any questions, please feel free to contact us:
    - [Chris Tran](https://www.linkedin.com/in/chriskhanhtran/)
    - [Chenxi Tao](https://www.linkedin.com/in/chenxitao/)
    - [Pin Li](https://www.linkedin.com/in/lpin/)
    - [Ruiling Shen](https://www.linkedin.com/in/ruiling-shen/)
    - [Jiawen Liang](https://www.linkedin.com/in/davinaliang/)
    """)
    
#####################################################################################
def data_page():
    st.sidebar.markdown('[Download Data](https://github.com/chriskhanhtran/fico-default-risk/blob/master/data/heloc_dataset_v1.csv)')
    st.markdown('# Data Dictionary')
    st.write('**Feature Explanations**')
    data_dict_1 = pd.read_csv('data/data_dict_1.csv')
    st.dataframe(data_dict_1)
    
    st.write('**MaxDelq Tablen**')
    st.write('MaxDelq2PublicRecLast12M')
    data_dict_2 = pd.read_csv('data/data_dict_MaxDelq2PublicRecLast12M.csv')
    st.table(data_dict_2)
    
    st.write('MaxDelqEver')
    data_dict_3 = pd.read_csv('data/data_dict_MaxDelqEver.csv')
    st.table(data_dict_3)    

    st.write('**Special Values**')
    data_dict_4 = pd.read_csv('data/data_dict_SpecialValues.csv')
    st.table(data_dict_4)

#####################################################################################
def model_page():
    st.markdown('# FICO Default Risk Prediction')
    ### Load model
    model_names = ['logit', 'rfc', 'svc', 'lgbm']

    @st.cache(show_spinner=False)
    def load_model(model_names):
        models = []
        for model_name in model_names:
            filename = 'models/' + model_name + '.joblib.pkl'
            model = joblib.load(filename)
            models.append(model)
        return models

    models = load_model(model_names)
    logit, rfc, svc, lgbm = models

    ### Select model
    selected_model = st.selectbox('Model: ', ['Light Gradient Boosting Model',
                                              'Logistic Regression',
                                              'Random Forest',
                                              'Support Vectors Machine'])

    if selected_model == 'Logistic Regression':
        model = logit
    elif selected_model == 'Random Forest':
        model = rfc
    elif selected_model == 'Support Vectors Machine':
        model = svc
    elif selected_model == 'Light Gradient Boosting Model':
        model = lgbm

    ### Input values
    features = {}
    st.sidebar.markdown("## Variables:")
    st.sidebar.markdown("---")

    st.sidebar.markdown("**External Risk Estimate**")
    features['ExternalRiskEstimate'] = st.sidebar.slider('ExternalRiskEstimate', -9, 94, 67)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Trade Open Time**")
    features['MSinceOldestTradeOpen'] = st.sidebar.slider('MSinceOldestTradeOpen', -9, 803, 184)
    features['MSinceMostRecentTradeOpen'] = st.sidebar.slider('MSinceMostRecentTradeOpen', -9, 383, 8)
    features['AverageMInFile'] = st.sidebar.slider('AverageMInFile', -9, 383, 73)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Satisfactory Trades**")
    features['NumSatisfactoryTrades'] = st.sidebar.slider('NumSatisfactoryTrades', -9, 79, 19)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Trade Frequency**")
    features['NumTrades60Ever2DerogPubRec'] = st.sidebar.slider('NumTrades60Ever2DerogPubRec', -9, 19, 0)
    features['NumTrades90Ever2DerogPubRec'] = st.sidebar.slider('NumTrades90Ever2DerogPubRec', -9, 19, 0)
    features['NumTotalTrades'] = st.sidebar.slider('NumTotalTrades', -9, 104, 20)
    features['NumTradesOpeninLast12M'] = st.sidebar.slider('NumTradesOpeninLast12M', -9, 19, 1)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Delinquency**")
    features['PercentTradesNeverDelq'] = st.sidebar.slider('PercentTradesNeverDelq', -9, 100, 86)
    features['MSinceMostRecentDelq'] = st.sidebar.slider('MSinceMostRecentDelq', -9, 83, 6)
    features['MaxDelq2PublicRecLast12M'] = st.sidebar.slider('MaxDelq2PublicRecLast12M', -9, 9, 5)
    features['MaxDelqEver'] = st.sidebar.slider('MaxDelqEver', -9, 8, 5)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Installment**")
    features['PercentInstallTrades'] = st.sidebar.slider('PercentInstallTrades', -9, 100, 32)
    features['NetFractionInstallBurden'] = st.sidebar.slider('NetFractionInstallBurden', -9, 471, 39)
    features['NumInstallTradesWBalance'] = st.sidebar.slider('NumInstallTradesWBalance', -9, 23, 1)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Inquiry**")
    features['MSinceMostRecentInqexcl7days'] = st.sidebar.slider('MSinceMostRecentInqexcl7days', -9, 24, 0)
    features['NumInqLast6M'] = st.sidebar.slider('NumInqLast6M', -9, 66, 0)
    features['NumInqLast6Mexcl7days'] = st.sidebar.slider('NumInqLast6Mexcl7days', -9, 66, 0)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Revolving Balance**")
    features['NetFractionRevolvingBurden'] = st.sidebar.slider('NetFractionRevolvingBurden', -9, 232, 32)
    features['NumRevolvingTradesWBalance'] = st.sidebar.slider('NumRevolvingTradesWBalance', -9, 32, 3)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**Utilization**")
    features['NumBank2NatlTradesWHighUtilization'] = st.sidebar.slider('NumBank2NatlTradesWHighUtilization', -9, 18, 0)
    st.sidebar.markdown("---")

    st.sidebar.markdown("**TradeWBalance**")
    features['PercentTradesWBalance'] = st.sidebar.slider('PercentTradesWBalance', -9, 100, 62)

    ### Load a sample
    X = pd.read_csv('data/sample.csv')
    X.iloc[:, :] = 0

    ### Preprocess X
    for feat_name in features.keys():
        # One-hot encode for 'MaxDelq2PublicRecLast12M' and 'MaxDelqEver'
        if feat_name in ['MaxDelq2PublicRecLast12M', 'MaxDelqEver']:
            if features[feat_name] > 0:
                X[feat_name+'_'+str(features[feat_name])] = 1
        else:
            # One-hot encode for negative values
            if features[feat_name] in [-9, -8, -7]:
                X[feat_name] = np.nan
                X[feat_name+'_'+str(features[feat_name])] = 1
            else:
                X[feat_name] = features[feat_name]

    ### Show model's evaluation
    def show_evaluation():
        if st.checkbox("Show Model's Evaluation"):
            if selected_model == 'Logistic Regression':
                st.write('**Logistic Regression** is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 or 0. In other words, the logistic regression model predicts P(Y=1) as a function of X.')
                st.write('**The model is evaluated on a test set randomly selected from 10% of the entire dataset.*')
                st.write('**AUC:** 0.7996')
                st.write('**Accuracy:** 73.04%')
                image = Image.open('img/logit.png')
                st.image(image, use_column_width=False)

            elif selected_model == 'Random Forest':
                st.write('**Random Forests** algorithm is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.')
                st.write('**The model is evaluated on a test set randomly selected from 10% of the entire dataset.*')
                st.write('**AUC:** 0.7981')
                st.write('**Accuracy:** 74.00%')
                image = Image.open('img/rfc.png')
                st.image(image, use_column_width=False)

            elif selected_model == 'Support Vectors Machine':
                st.write('**Support Vector Machine (SVM)** constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (functional margin).')
                st.write('**The model is evaluated on a test set randomly selected from 10% of the entire dataset.*')
                st.write('**AUC:** 0.7919')
                st.write('**Accuracy:** 72.16%')
                image = Image.open('img/svc.png')
                st.image(image, use_column_width=False)

            elif selected_model == 'Light Gradient Boosting Model':
                st.write('**LightGBM** is a new gradient boosting tree framework, which is highly efficient and scalable and can support many different algorithms including GBDT, GBRT, GBM, and MART. LightGBM is evidenced to be several times faster than existing implementations of gradient boosting trees, due to its fully greedy tree-growth method and histogram-based memory and computation optimization.')
                st.write('**The model is evaluated on a test set randomly selected from 10% of the entire dataset.*')
                st.write('**AUC:** 0.8044')
                st.write('**Accuracy:** 73.42%')
                image = Image.open('img/lgbm.png')
                st.image(image, use_column_width=False)
            
            st.write("""
            **Top 5 important features:**
            - ExternalRiskEstimate
            - AverageMInFile
            - NumSatisfactoryTrades
            - MSinceMostRecentInqexcl7days
            - NetFractionRevolvingBurden
            """)
            
    ### Show prediction
    def show_prediction():
        #if st.button("Run Model"):
        pred = model.predict_proba(X)[0, 1]
        st.markdown(f'## Risk of Default: {pred*100:.2f} %')
        
    show_prediction()    
    show_evaluation()
 
#####################################################################################
page = st.sidebar.selectbox('Page', ['Models', 'Data', 'Documents'])
if page == 'Models':
    model_page()
elif page == 'Data':
    data_page()
elif page == 'Documents':
    document_page()


