import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

st.title('Email/SMS Spam Classifier')


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

input_message = st.text_area("Enter your message")
if st.button('Predict'):

    # Preprocess
    transform_message = transform_text(input_message)
    # Vectorization
    vector_input = tfidf.transform([transform_message])
    # Predict
    result = model.predict(vector_input)[0]
    # Result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')
