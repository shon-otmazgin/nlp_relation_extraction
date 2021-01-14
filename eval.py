import sys
from utils import read_annotations

gold_annotations = read_annotations(sys.argv[1])
pred_annotations = read_annotations(sys.argv[2])

TP, FP = 0, 0

for sent_id, annotations in pred_annotations.items():
    for pred_ann in annotations:
        per, _, org = pred_ann
        T = False
        for glod_ann in gold_annotations[sent_id]:
            if (per in glod_ann[0] or glod_ann[0] in per) and (org in glod_ann[2] or glod_ann[2] in org):
                T = True
                break
        if T:
            TP += 1
        else:
            FP += 1

P = TP / (TP+FP)

TP, FN = 0, 0
for sent_id, annotations in gold_annotations.items():
    for gold_ann in annotations:
        per, _, org = gold_ann
        T = False
        for pred_ann in pred_annotations[sent_id]:
            if (per in pred_ann[0] or pred_ann[0] in per) and (org in pred_ann[2] or pred_ann[2] in org):
                T = True
                break
        if T:
            TP += 1
        else:
            FN += 1

R = TP / (TP+FN)
print(f'Precision: {P:.3f}')
print(f'Recall:    {R:.3f}')
print(f'F1:        {((2*R*P)/(P+R)):.3f}')