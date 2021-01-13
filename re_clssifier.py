from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


with open('pickles/train', 'rb') as f:
    df, y = pickle.load(f)


kf = KFold(n_splits=10, random_state=42, shuffle=True)
X = df.to_numpy()
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.01).fit(X_train, y_train)
    svm_clf = SVC(kernel='linear', random_state=42, C=0.01).fit(X_train, y_train)

    y_pred = lr_clf.predict(X_test)
    # print(f'y_test: {y_test}')
    # print(f'y_pred: {y_pred}')
    print(f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')
    # for idx in df.iloc[test_index[y_test != y_pred]].index:
    #     print(idx)


with open('pickles/dev', 'rb') as f:
    X_test, y_test = pickle.load(f)

# y_pred = lr_clf.predict(X)
# print('TRAIN')
# print(f'P={precision_score(y, y_pred):.3f}, R={recall_score(y, y_pred):.3f}, F={f1_score(y, y_pred):.3f}')
#
lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.01).fit(df, y)
y_pred = lr_clf.predict(X_test)
print('DEV')
print(f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')
#
# y_pred = svm_clf.predict(X)
# print('TRAIN')
# print(f'P={precision_score(y, y_pred):.3f}, R={recall_score(y, y_pred):.3f}, F={f1_score(y, y_pred):.3f}')
#
# y_pred = svm_clf.predict(X_test)
# print('DEV')
# print(f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')

# X_test['y'] = y_test
# X_test['y_pred'] = y_pred
#
# # precision calc
# # print(X_test[(X_test['y_pred'] == 1) & (X_test['y'] == 1)].shape[0] / X_test[X_test['y_pred'] == 1].shape[0])
#
# for idx in X_test[(X_test['y_pred'] == 1) & (X_test['y'] == 1)].head().index:
#     print(idx)