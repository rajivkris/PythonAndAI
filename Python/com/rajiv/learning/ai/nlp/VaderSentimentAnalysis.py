import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

nltk.download('vader_lexicon')

text = "This is a awesome good movie!!!"
sid = SentimentIntensityAnalyzer()

df = pd.read_csv('./TextFiles/amazonreviews.tsv', sep='\t')

df.dropna(inplace=True)
blanks = []
for i, lb, rv in df.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

for i in blanks:
    df.drop(i, inplace=True)

df['scores'] = df['review'].apply(lambda review: sid.polarity_scores(review))
df['compound'] = df['scores'].apply(lambda d: d['compound'])
df['compound_label'] = df['scores'].apply(lambda score: 'pos' if score['compound'] >= 0 else 'neg')

ac_score = accuracy_score(df['label'], df['compound_label'])
con_metrics = confusion_matrix(df['label'], df['compound_label'])
class_report = classification_report(df['label'], df['compound_label'])

print(ac_score)
print(con_metrics)
print(class_report)