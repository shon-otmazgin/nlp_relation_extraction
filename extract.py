import sys
import pickle

from features_extraction import build_df
from rules import rule_retired, rule_org_s

with open('trained_model', 'rb') as f:
    lr_clf, V, vocab = pickle.load(f)

test_df, V, vocab = build_df(file=sys.argv[1], V=V, vocab=vocab)
print(f'Test size: {test_df.shape}')

test_df['y_pred'] = lr_clf.predict(test_df)

output_file = sys.argv[2]
with open(output_file, 'w', encoding="utf8") as f:
    for idx in test_df[test_df['y_pred'] == 1].index:
        sent_id, person, org, sent = idx
        # if rule_retired(person=person, org=org, sent=sent):
        #     continue
        # if rule_org_s(person=person, org=org, sent=sent):
        #     continue
        f.write(f'{sent_id}\t{person}\tWork_For\t{org}\t{sent}\n')

print(f'Relation extracted to: {output_file}')