import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    #1
    text = text.lower()
    #2
    text = nltk.word_tokenize(text)
    #3
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    #4
    text = y[:] # cloneing 
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    # 5
    text = y[:] # cloneing 
    y.clear()  
    
    for i in text:
        y.append(ps.stem(i))    
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

imput_sms = st.text_input('Enter the message')

if st.button('predict'):

  # 1. preprocess
  transformed_sms = transform_text(imput_sms)
  # 2. vectorize
  vector_input = tfidf.transform([transformed_sms])
  # 3. predict
  result = model.predict(vector_input)[0]
   # 4. display
  if result == 1:
     st.header('Spam')
  else :
     st.header('Not Spam')

