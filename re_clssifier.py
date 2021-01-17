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


# train('train')
# dev('train', 'dev')

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def double_workplace(annotations_file):
    output_file_name = annotations_file.split('.')
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        data = [line.strip().split('\t')[0:5] for line in in_f]



    df = pd.DataFrame(data, columns=['sentid', 'per', 'rel', 'org', 'sent'])
    gby = df.groupby(by=['sentid', 'per'])
    for g in gby.groups:
        db_work_df = gby.get_group(g)
        if db_work_df.shape[0] > 1:
            sent = db_work_df['sent'].values[0]
            print(sent)
            for i in db_work_df.index:
                per, org = db_work_df.loc[i]['per'], db_work_df.loc[i]['org']
                print(per, org)
                template = f'{per} , a {org}'
                if template in sent:
                    print(True)
                # tokens = sent.split()
                # # sent.rfind(per)
                # # sent.rfind(org)
                # wb = sent[sent.rfind(per):sent.index(org)] if sent.rfind(per) < sent.index(org) else sent[sent.rfind(org):sent.index(per)]
                # print(sent)
                # print(per)
                # print(wb)
                # print(org)
                # print('------------')
            print()

double_workplace('predicted_relation_train.txt')