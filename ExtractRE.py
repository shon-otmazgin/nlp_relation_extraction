import sys
import pickle

from features_extraction import build_df

with open(sys.argv[2], 'rb') as f:
    model = pickle.load(f)

with open(sys.argv[3], 'rb') as f:
    V = pickle.load(f)

test_df, V = build_df(file=sys.argv[1], V=V, vectors_file=sys.argv[4])
print(f'Test size: {test_df.shape}')

test_df['y_pred'] = model.predict(test_df)

with open('predicted_relation.txt', 'w', encoding="utf8") as f:
    for idx in test_df[test_df['y_pred'] == 1].index:
        sent_id, person, org, sent = idx
        f.write(f'{sent_id}\t{person}\tWork_For\t{org}\t{sent}\n')

print(f'Relation extracted to: predicted_relation.txt')