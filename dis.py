import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
        
        def convert_lower(text) :
             text = text.lower()
             return text
        df['text'] = df['text'].apply(convert_lower)
        def remove_stopwords(text) :
             new_list = []
             words = word_tokenize(text)
             sw = stopwords.words('english')
             sw.extend(["'ve",'discomfort'])
             for word in words:
                  if word not in sw:
                       new_list.append(word)
             return " ".join(new_list)
        df['text'] = df['text'].apply(remove_stopwords)
        def lemma(text):
             lemmatize_text = []
             words = word_tokenize(text)
             for word in words :
                  lemmatize_text.append(WordNetLemmatizer().lemmatize(word))
             return " ".join(lemmatize_text)
        df['text'] = df['text'].apply(lemma)
        X = df['text']
        y = df['label']
        X_count_vect = count_vect.fit_transform(X)
        #smote = SMOTE()
        #X_resampled, y_resampled = smote.fit_resample(X_count_vect,y)
        symptoms = convert_lower(symptoms)
        symptoms = remove_stopwords(symptoms)
        symptoms = lemma(symptoms)
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
        if i in pred == 0:
             st.chat_message("assistant").write("""You are infected with malaria you need to see the doctor for 
                             medicine prescription
                             """)
        elif i in pred == 1:
             st.chat_message("assistant").write("""You are infected with typhoid you need to see the doctor for 
                             medicine prescription
                             """)
        else :
             st.chat_message("assistant").write("""The diagnosis isnt you need to see the doctor for 
                             medicine prescription
                             """)
        st.chat_message("assistant").write(f"Echo: {pred}, {pred_proba}")