import streamlit as st
import numpy as np
import pandas as pd
import textblob
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
import re

nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

analyzer = SentimentIntensityAnalyzer()

st.title("RE-VEU")
st.header("Customer feedback is very Important. \n Re-Vue analysis your data and helps you to get better insights about reviews from your customers.")
st.markdown("Let's get started, please upload a file")


uploaded_file = st.file_uploader(label="Upload a File",
                                 type=['csv','xlsx'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('## Dataset')
    st.dataframe(data)

    st.write('Select the column you want to analyse')
    col = st.selectbox("Column Name",(data.columns))

    if st.button("Click for Results") :
        

        def CleanText(df):
            tempText = []
            for line in df:
                text = line
                text = re.sub(r"[^a-zA-Z0-9]+", ' ' ,text.lower())
                tempText.append(text)

            return tempText
        
        text = data[col].apply(str)
        data['Text'] = CleanText(text)
        data['Reviews'] = CleanText(text)

        def short_text(df):
            tempText = []
            for line in df:
                text = line
                stopword = set(stopwords.words('english'))
                words =  word_tokenize(text)
                text = [w for w in words if w not in stopword]
                text = ' '.join(text)
                tempText.append(text)
            return tempText

        data['Reviews'] = short_text(data['Reviews'])

        def sentiment(df):
            tempList = []
            for line in df:
                text = line
                check = analyzer.polarity_scores(text)
                if check['compound'] >= 0.05:
                    tempList.append("Positive")
                elif check['compound'] <= - 0.05:
                    tempList.append("Negative")
                else:
                    tempList.append("Neutral")

            return tempList
        
        data['Sentiment'] = sentiment(data['Reviews'])

        def service(df):
            service = []
            for line in df:
                words = word_tokenize(line)
                text = nltk.pos_tag(words)
                tempdict = dict(text)
                vals = list(tempdict.values())
                prouns = ['VB','VBD','VBG','VBN']
                templis = []
                for i in vals:
                    for j in prouns:
                            if i == j:
                                templis.append('Yes')
                            else:
                                templis.append('No')
                service.append(len(list(filter(lambda x: x == 'Yes', templis))))
            return service

        def product(df):
            product = []
            for line in df:
                words = word_tokenize(line)
                text = nltk.pos_tag(words)
                tempdict = dict(text)
                vals = list(tempdict.values())
                strs = ' '.join([str(elem) for elem in vals])
                prouns = ['NN','NNS']
                templis = []
                for i in vals:
                    for j in prouns:
                        if i == j:
                            templis.append('Yes')
                        else:
                            templis.append('No')

                product.append(len(list(filter(lambda x: x == 'Yes', templis))))

            return product

        def staff(df):
            staff = []
            for line in df:
                words = word_tokenize(line)
                text = nltk.pos_tag(words)
                tempdict = dict(text)
                vals = list(tempdict.values())
                strs = ' '.join([str(elem) for elem in vals])
                prouns = ['NNP','NNPS','PRP','PRP$']
                templis = []
                for i in vals:
                    for j in prouns:
                        if i == j:
                            templis.append('Yes')
                        else:
                            templis.append('No')
                staff.append(len(list(filter(lambda x: x == 'Yes', templis))))

            return staff

        data['Service'] = service(data["Text"])
        data['Product'] = product(data["Text"])
        data['Staff'] = staff(data["Text"])

        final_df = data[['Reviews','Sentiment','Service','Product','Staff']]

        positive = len(data[data['Sentiment'] == 'Positive'])
        neutral = len(data[data['Sentiment'] == 'Neutral'])
        negative = len(data[data['Sentiment'] == 'Negative'])

        st.write(f" Positive Reviews : {positive} | Neutral Reviews : {neutral} | Negative Reviews : {negative}")
        st.dataframe(final_df)

        st.download_button(
            label="Download data as CSV",
            data= final_df.to_csv().encode("utf-8"),
            file_name='data.csv',
            mime='text/csv',
        )