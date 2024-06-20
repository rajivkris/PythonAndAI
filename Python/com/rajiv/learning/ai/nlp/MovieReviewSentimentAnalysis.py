import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('./TextFiles/moviereviews.tsv', sep='\t')
print(df.head())

df.dropna(inplace=True)
blanks = []

for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

df.drop(blanks, inplace=True)

sid = SentimentIntensityAnalyzer()
df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda score_dict: score_dict['compound'])

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >= 0 else 'neg')
print(accuracy_score(df['label'], df['comp_score']))
print(confusion_matrix(df['label'], df['comp_score']))
print(classification_report(df['label'], df['comp_score']))