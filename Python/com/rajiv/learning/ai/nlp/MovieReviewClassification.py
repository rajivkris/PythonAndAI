import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

# Read the data
df = pd.read_csv('./TextFiles/moviereviews2.tsv', sep='\t')
df.dropna(inplace=True)
print(df.head())
for i, label, text in df.itertuples():
    if text.isspace():
        df.drop(i, inplace=True)

X = df['review']
y= df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)
predictions = clf.predict(tfidf_vectorizer.transform(X_test))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
print(accuracy_score(y_test, predictions))
print(clf.predict(tfidf_vectorizer.transform(['This movie was awesome!'])))
print(clf.predict(tfidf_vectorizer.transform(['This movie was terrible'])))

