import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.metrics as metrics

# Read the data
df = pd.read_csv('./TextFiles/smsspamcollection.tsv', sep='\t')
print(df.head())

X = df[['length', 'punct']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr_model = LogisticRegression(solver='lbfgs')
lr_model.fit(X_train, y_train)

predictions=lr_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(pd.DataFrame(metrics.confusion_matrix(y_test, predictions), index=['ham', 'spam'], columns=['ham', 'spam']))
print(metrics.accuracy_score(y_test, predictions))
print(metrics.precision_score(y_test, predictions, pos_label='ham'))
print(metrics.classification_report(y_test, predictions))

svc_model = SVC(gamma='auto')
svc_model.fit(X_train, y_train)
predictions = svc_model.predict(X_test)
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.classification_report(y_test, predictions))
