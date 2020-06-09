import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('E:\\CC_Internship\\news.csv')
print(df.shape)
print(df.head())

labels = df.label
labels.head()

x_train, x_test, y_train, y_test = train_test_split(df['text'], labels, test_size=0.2, random_state=7)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train = vectorizer.fit_transform(x_train)
tfidf_test = vectorizer.transform(x_test)

pc = PassiveAggressiveClassifier(max_iter=50)
pc.fit(tfidf_train, y_train)

y_pred = pc.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print()

print(f'Accuracy: {score}')
print()

# confusion matrix
confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL']))

'''
Output :

(6335, 4)
   Unnamed: 0  ... label
0        8476  ...  FAKE
1       10294  ...  FAKE
2        3608  ...  REAL
3       10142  ...  FAKE
4         875  ...  REAL

[5 rows x 4 columns]

Accuracy: 0.925808997632202

Confusion Matrix :
[[588  50]
 [ 44 585]]

Process finished with exit code 0

'''