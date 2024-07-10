import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#from imblearn.over_sampling import SMOTE
tab_1, tab_2,tab_3 = st.tabs(['Diagnose','Health Information'])

with open('mrforest.pkl', 'rb') as file:
        forest = pickle.load(file)
with open('mlogreg.pkl', 'rb') as file:
        logreg = pickle.load(file)
count_vect = TfidfVectorizer(stop_words= 'english')

option = st.sidebar.selectbox("Choose the action to perform in the application",('Diagnose','Health Information'))
    
if option == 'Diagnose' :

    df = pd.read_csv("diseases.csv")
    symptoms = tab_1.chat_input("How do you feel like")
    if symptoms :
        tab_1.chat_message("user").write(symptoms)
        
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
        for i in range(140) :
            if i in df_count :
                if i != 0 :
                    DF.append(i)
                else :
                    DF.append(0)
            else :
                DF.append(0)
        df_conv = pd.DataFrame([DF])  

        pred = forest.predict(df_conv)
        pred_proba = forest.predict_proba(df_conv) 
        if i in pred == "Malaria":
             tab_1.chat_message("assistant").write("""You are infected with malaria you need to see the doctor for 
                             medicine prescription
                             """)
             tab_2.chat_message("assistant").write("""
          The treatment for malaria typically involves taking antimalarial medications, as prescribed by the doctor,
           depending on the specific type of malaria and its resistance patterns.
           It is important to start treatment as soon as possible after diagnosis to prevent complications and ensure a quick recovery.
           In severe cases, hospitalization may be necessary for intravenous medications and supportive care.
           Additionally, maintaining hydration, getting adequate rest,
          and following up with healthcare providers for monitoring and further evaluation are essential components of the treatment plan.
                                                   
          """) 
        #elif i in pred == "Typhoid":
        else :
             tab_1.chat_message("assistant").write("""You are infected with typhoid you need to see the doctor for 
                             medicine prescription
                             """)
             tab_2.chat_message("assistant").write("""
          The primary cure for typhoid fever involves antibiotics,That will be prescribed by the doctor,
          .Additionally,supportive treatments such as hydration, rest, and proper nutrition are important to aid recovery.
          In addition to taking antibiotics, it is crucial to stay well-hydrated by drinking plenty of fluids to prevent dehydration caused by fever and diarrhea.
           Getting ample rest is essential to help your body fight off the infection and recover more quickly.
           Maintaining proper nutrition by eating light, easily digestible foods such as soups, fruits, and boiled vegetables supports the immune system.

          Practicing good personal hygiene, including regular handwashing with soap and water, helps prevent the spread of the infection to others.
           It is important to consume only safe, well-cooked food and drink purified or bottled water to avoid re-infection.
           Attending follow-up appointments with your healthcare provider ensures the infection is completely cleared and monitors for any potential complications.
          Finally, consider getting vaccinated against typhoid fever if you live in or plan to travel to areas where the disease is common.
                                                   
          """)
        #else :
             #tab_1.chat_message("assistant").write("""The diagnosis isnt you need to see the doctor for 
                             
              #               """)
             #tab_2.chat_message("assistant").write("""
          #The diagnosis is neither malaria nor typhoid, you are advised to contact the doctor for further test.                                       
         # """)
        #tab_1.chat_message("assistant").write(f"Echo: {pred}, {pred_proba}")

else :
     info_option = st.sidebar.selectbox("Choose the health information needed",('Malaria','Typhoid'))
     if info_option == "Malaria" :
          tab_2.title("Malaria Health Information")
          tab_2.write(
          """
          Malaria is a very old disease, with references found in ancient Chinese, Greek, and Roman texts,
          and for a long time, people believed it was caused by bad air from swamps.
          In the late 1800s, a scientist named Charles Louis Alphonse Laveran discovered that malaria is actually caused by tiny parasites.
          Later, Ronald Ross found that these parasites are spread by mosquitoes, specifically female Anopheles mosquitoes,
          which bite humans and pass the parasites into their bloodstream.

          Malaria is caused by parasites called Plasmodium,
          with five types that can infect humans: Plasmodium falciparum, P. vivax, P. malariae, P. ovale, and P. knowlesi.
          These parasites enter the body when an infected mosquito bites a person, travel to the liver to grow and multiply,
          and then move into the bloodstream to infect red blood cells.

          The symptoms of malaria usually start 10-15 days after an infected mosquito bite.
          Common symptoms include high fever, severe chills, persistent headaches, heavy sweating,
          extreme tiredness, aching muscles and joints, nausea and vomiting,
          stomach pain, frequent loose bowel movements, low red blood cell count (anemia) causing paleness and shortness of breath,
          and yellowing of the skin and eyes (jaundice). In severe cases, malaria can cause confusion, seizures, and organ failure, which can be life-threatening.

          To prevent malaria, several steps can be taken:
          use insect repellents containing DEET or picaridin to keep mosquitoes away,
          sleep under insecticide-treated mosquito nets to avoid bites at night,
          wear long sleeves and pants especially during dawn and dusk when mosquitoes are most active,
          and spray the inside of homes with insecticides to kill mosquitoes. Additionally, eliminate standing water around your home where mosquitoes can breed,
          take preventive antimalarial medications if you are traveling to a high-risk area, and consider getting vaccinated with the RTS,S/AS01 (Mosquirix) vaccine,
          especially if you live in or are traveling to regions where malaria is common.

     """ )
     else :
          tab_2.title("Typhoid Information")
          tab_2.write(
          """
               Typhoid fever is a disease that has been known for many centuries.
               It was first clearly described in the early 19th century, but the bacteria that cause it, Salmonella Typhi, were discovered by Karl Joseph Eberth in 1880.
               Before the development of modern sanitation and antibiotics, typhoid fever caused many outbreaks and was a significant cause of illness and death.

     Cause
     Typhoid fever is caused by the bacteria Salmonella Typhi. People get infected by consuming food or water that has been contaminated with the feces of an infected person.
     This can happen when proper hygiene practices, like washing hands with soap, are not followed, or when sewage contaminates drinking water.

     Symptoms
     Symptoms of typhoid fever usually appear 1-3 weeks after exposure to the bacteria. Common symptoms include:

     Fever: A high, sustained fever that can reach up to 104°F (40°C).
     Weakness: Feeling very weak and tired.
     Stomach pain: Pain or discomfort in the abdominal area.
     Headache: Persistent pain in the head.
     Loss of appetite: Not feeling hungry and eating less than usual.
     Diarrhea or constipation: Some people may have diarrhea, while others may experience constipation.
     Rash: Some people may develop a rash of flat, rose-colored spots.
     If not treated, typhoid fever can become very serious, leading to complications such as intestinal bleeding or perforation, which can be life-threatening.

     Preventive Measures
     To prevent typhoid fever, follow these steps:

     Good Hygiene: Always wash your hands thoroughly with soap and water, especially before eating and after using the toilet.
     Safe Drinking Water: Drink only boiled or bottled water. Avoid ice cubes unless you are sure they are made from safe water.
     Proper Food Handling: Eat food that is thoroughly cooked and still hot. Avoid raw fruits and vegetables unless you can peel them yourself.
     Avoid Street Food: Be cautious with food from street vendors, as it might not be prepared under hygienic conditions.
     Vaccination: Get vaccinated against typhoid fever, especially if you are traveling to areas where the disease is common.
     There are two main types of vaccines: an injection given as a single dose and an oral vaccine taken in multiple doses over several days.
     Sanitation: Ensure proper sanitation practices are in place to prevent contamination of food and water sources.
     By following these preventive measures, you can significantly reduce your risk of getting typhoid fever.

     """ 
          )