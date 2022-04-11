from flask import Flask,request
app = Flask(__name__)

@app.route('/',methods=['POST'])
def ml_model():
    import pandas as pd
    import numpy as np
    datafile = request.files.get('file')
    df = pd.read_csv(datafile)
    df = df.drop('Developer Reply',axis=1)
    df = df.dropna()
    def clean_text(text):
        text = re.sub('[^A-Za-z]+',' ', text)
        return text
    df['Cleaned_reviews'] = df['Text'].apply(clean_text)
    from textblob import TextBlob
    def getSubjectivity(review):
        return TextBlob(review).sentiment.subjectivity
    def getPolarity(review):
        return TextBlob(review).sentiment.polarity

    def analysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'  
    df['Subjectivity'] = df['Cleaned_reviews'].apply(getSubjectivity)
    df['Polarity'] = df['Cleaned_reviews'].apply(getPolarity)
    df['Analysis'] = df['Polarity'].apply(analysis)
    res = df.Cleaned_reviews[(df['Analysis']=='Positive')&(df['Star']==1)]
    return res


if __name__ == "__main__":
   app.run(debug=True)
