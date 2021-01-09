from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import pickle
pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


with open('train.pkl', 'rb') as f:
    X, y = pickle.load(f)

with open('dev.pkl', 'rb') as f:
    X_test, y_test = pickle.load(f)

lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.1).fit(X, y)
svm_clf = SVC(kernel='linear', random_state=42, C=0.1).fit(X, y)

y_pred = lr_clf.predict(X)
print('TRAIN')
print(f'P={precision_score(y, y_pred):.3f}, R={recall_score(y, y_pred):.3f}, F={f1_score(y, y_pred):.3f}')

y_pred = lr_clf.predict(X_test)
print('DEV')
print(f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')

y_pred = svm_clf.predict(X)
print('TRAIN')
print(f'P={precision_score(y, y_pred):.3f}, R={recall_score(y, y_pred):.3f}, F={f1_score(y, y_pred):.3f}')

y_pred = svm_clf.predict(X_test)
print('DEV')
print(f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')

X_test['y'] = y_test
X_test['y_pred'] = y_pred

# precision calc
# print(X_test[(X_test['y_pred'] == 1) & (X_test['y'] == 1)].shape[0] / X_test[X_test['y_pred'] == 1].shape[0])

for idx in X_test[(X_test['y_pred'] == 1) & (X_test['y'] == 1)].head().index:
    print(idx)