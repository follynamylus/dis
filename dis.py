import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from imblearn.over_sampling import SMOTE
tab_2, tab_1,tab_3 = st.tabs(['Diagnose','Information About Diagnosis','Health Information'])

with open('rforest.pkl', 'rb') as file:
        logreg = pickle.load(file)
count_vect = CountVectorizer(stop_words= 'english')

option = st.sidebar.selectbox("Choose the action to perform in the application",('Diagnose','Health Information'))
    
if option == 'Diagnose' :

    df = pd.read_csv("diseases.csv")
    symptoms = st.chat_input("How do you feel like")
    if symptoms :
        st.chat_message("user").write(symptoms)
        X = df['text']
        y = df['label']
        X_count_vect = count_vect.fit_transform(X)
        #smote = SMOTE()
        #X_resampled, y_resampled = smote.fit_resample(X_count_vect,y)
        data_vec = count_vect.fit_transform([symptoms])
        df_count = pd.DataFrame(X_count_vect)
        DF = []
        for i in range(378) :
            if i in df_count :
                if i != 0 :
                    DF.append(i)
                else :
                    DF.append(0)
            else :
                DF.append(0)
        df_conv = pd.DataFrame([DF])  

        pred = logreg.predict(df_conv)
        pred_proba = logreg.predict_proba(df_conv)      

        st.chat_message("assistant").write(f"Echo: {pred}, {pred_proba}")