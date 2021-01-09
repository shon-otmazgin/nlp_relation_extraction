from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd

train_df = pd.read_pickle('train_df.pkl')
print(f'dataframe size from pickle: {train_df.shape}')
print(f'response varibale (y) counts:\n{train_df["y"].value_counts()}')

dev_df = pd.read_pickle('dev_df.pkl')
print(f'dataframe size from pickle: {dev_df.shape}')
print(f'response varibale (y) counts:\n{dev_df["y"].value_counts()}')

lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.001).fit(train_df.drop(columns=['y']), train_df['y'])
svm_clf = SVC(kernel='linear', random_state=42, C=0.01).fit(train_df.drop(columns=['y']), train_df['y'])


y_pred = lr_clf.predict(train_df.drop(columns=['y']))
print(classification_report(train_df['y'], y_pred))

y_pred = lr_clf.predict(dev_df.drop(columns=['y']))
print(classification_report(dev_df['y'], y_pred))

y_pred = svm_clf.predict(train_df.drop(columns=['y']))
print(classification_report(train_df['y'], y_pred))

y_pred = svm_clf.predict(dev_df.drop(columns=['y']))
print(classification_report(dev_df['y'], y_pred))

# df = pd.read_pickle('df.pkl')
# print(f'dataframe size from pickle: {df.shape}')
# print(f'response varibale (y) counts:\n{df["y"].value_counts()}')
#
# X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['y']), df['y'], test_size=0.33, random_state=42)
#
# lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.001).fit(X_train, y_train)
# svm_clf = SVC(kernel='linear', random_state=42, C=0.01).fit(X_train, y_train)
#
# y_pred = lr_clf.predict(X_train)
# print(classification_report(y_train, y_pred))
#
# y_pred = lr_clf.predict(X_test)
# print(classification_report(y_test, y_pred))
#
# y_pred = svm_clf.predict(X_train)
# print(classification_report(y_train, y_pred))
#
# y_pred = svm_clf.predict(X_test)
# print(classification_report(y_test, y_pred))