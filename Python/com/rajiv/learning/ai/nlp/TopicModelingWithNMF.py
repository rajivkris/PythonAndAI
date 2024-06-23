import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_csv('./TopicModeling/quora_questions.csv')
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(df['Question'])

nmf = NMF(n_components=20, random_state=42)
nmf.fit(dtm)

for i, topic in enumerate(nmf.components_):
    print(f'Top 20 words for topic {i}')
    print([tfidf.get_feature_names_out()[index] for index in topic.argsort()[-15:]])

topicResults = nmf.transform(dtm)
df['Topic'] = topicResults.argmax(axis=1)
print(df.head())
