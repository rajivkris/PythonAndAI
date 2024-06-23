import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv('./TopicModeling/npr.csv')

cv = CountVectorizer(max_df=0.9, min_df=2, stop_words='english')
dtm = cv.fit_transform(df['Article'])
featureNames = cv.get_feature_names_out()
print(len(featureNames))

lda = LatentDirichletAllocation(n_components=7, random_state=42)
lda.fit(dtm)

for i in range(len(lda.components_)):
    print(f'Topic {i}')
    for j in lda.components_[i].argsort()[-10:]:
        print(featureNames[j])
    print('\n')

topicResults = lda.transform(dtm)

df['Topic'] = topicResults.argmax(axis=1)
print(df.head())
