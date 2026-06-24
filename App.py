import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

ps = PorterStemmer()

def transform_text(text):
    # Lower case
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title('📧 Email & SMS Spam Detector')
st.caption('Enter a message below and our machine learning model will determine whether it is Spam or Ham.')

input_sms = st.text_area('Enter your Email or SMS message', height = 150)

if st.button('🔍 Detect Spam'):
    if not input_sms.strip():
        st.error('Please enter a message before clicking Detect Spam.')
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Probability
        prob = model.predict_proba(vector_input)

        spam_prob = prob[0][1]
        ham_prob = prob[0][0]

        # Display
        if result == 1:
            st.error('🚨 Spam Message Detected')
            st.metric("🎯 Confidence", f"{spam_prob:.2%}")
        else:
            st.success('✅ Legitimate Message')
            st.metric("🎯 Confidence", f"{ham_prob:.2%}")

st.caption('Try these examples:')
st.code('Congratulations! You won $10,000. Click here to claim your prize.')
st.code("Hey John, let's meet at 7 PM tomorrow.")

st.sidebar.title('About')

st.sidebar.info('''
This project uses:
- TF-IDF Vectorization
- Multinomial Naive Bayes
- NLTK for text preprocessing
- Streamlit for deployment''')