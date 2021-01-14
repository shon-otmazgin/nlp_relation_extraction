from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def train(pkl='train'):
    with open(pkl, 'rb') as f:
        df, y = pickle.load(f)

    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    X = df.to_numpy()
    F = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.01).fit(X_train,
                                                                                                              y_train)
        y_pred = lr_clf.predict(X_test)

        # for idx in df.iloc[test_index[(y_pred == 1) & (y_test == 0)]].index:
        for i, p in enumerate(y_pred):
            if p == 0:
                continue
            idx = df.iloc[[test_index[i]]].index[0]
            sentid, person, org, sent = idx
            tokens = sent.split()
            if 'retired' in tokens:
                p_idx = tokens.index(person.split()[-1])
                o_idx = tokens.index(org.split()[-1])
                r_idx = tokens.index('retired')
                if (p_idx < r_idx < o_idx) or (o_idx < r_idx < p_idx):
                    if r_idx + 2 >= o_idx:
                        y_pred[i] = 1
                        # print(p_idx, r_idx, o_idx)
        for idx in df.iloc[test_index[(y_pred == 1) & (y_test == 0)]].index:
            print(idx)
        print()
        print(
            f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')
        F.append(f1_score(y_test, y_pred))
    print(np.mean(F))


def dev(train_pkl, dev_pkl):
    with open(train_pkl, 'rb') as f:
        df, y = pickle.load(f)
    with open(dev_pkl, 'rb') as f:
        X_test, y_test = pickle.load(f)

    lr_clf = LogisticRegression(random_state=42, class_weight='balanced', solver='liblinear', C=0.01).fit(df, y)
    y_pred = lr_clf.predict(X_test)
    print('DEV')
    print(
        f'P={precision_score(y_test, y_pred):.3f}, R={recall_score(y_test, y_pred):.3f}, F={f1_score(y_test, y_pred):.3f}')


train('train')
dev('train', 'dev')