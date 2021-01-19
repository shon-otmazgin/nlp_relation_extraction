from utils import read_annotations
import random

def get_precision(gold_annotations, pred_annotations, entirety=False):
  per_miss = []
  TP, FP = 0, 0
  for sent_id, annotations in pred_annotations.items():
      for pred_ann in annotations:
          per, _, org = pred_ann
          T = False
          for glod_ann in gold_annotations[sent_id]:
              if entirety:
                  if per == glod_ann[0] and org == glod_ann[2]:
                      T = True
                      break
              else:
                  if (per in glod_ann[0] or glod_ann[0] in per) and (org in glod_ann[2] or glod_ann[2] in org):
                      T = True
                      break
          if T:
              TP += 1
          else:
              FP += 1
              per_miss.append((sent_id, per, org))
  return TP / (TP+FP), per_miss


def get_recall(gold_annotations, pred_annotations, entirety=False):
  recall_miss = []
  TP, FN = 0, 0
  for sent_id, annotations in gold_annotations.items():
      for gold_ann in annotations:
          per, _, org = gold_ann
          T = False
          for pred_ann in pred_annotations[sent_id]:
              if entirety:
                  if per == pred_ann[0] and org == pred_ann[2]:
                      T = True
                      break
              else:
                  if (per in pred_ann[0] or pred_ann[0] in per) and (org in pred_ann[2] or pred_ann[2] in org):
                      T = True
                      break
          if T:
              TP += 1
          else:
              FN += 1
              recall_miss.append((sent_id, per, org))
  return TP / (TP+FN), recall_miss


def eval(train_file):
    print('TRAIN')
    gold_annotations = read_annotations('../data/TRAIN.annotations')
    pred_annotations = read_annotations(train_file)

    print(f'Entities extracted as entirety')
    P, P_miss = get_precision(gold_annotations, pred_annotations, entirety=True)
    R, R_miss = get_recall(gold_annotations, pred_annotations, entirety=True)
    print(f'Precision: {P:.3f}')
    print(f'Recall:    {R:.3f}')
    print(f'F1:        {((2 * R * P) / (P + R)):.3f}')

    print()
    print(f'Entities extracted with "in" method')
    P, P_miss = get_precision(gold_annotations, pred_annotations, entirety=False)
    R, R_miss = get_recall(gold_annotations, pred_annotations, entirety=False)
    print(f'Precision: {P:.3f}')
    print(f'Recall:    {R:.3f}')
    print(f'F1:        {((2 * R * P) / (P + R)):.3f}')

    print(f'precision miss:')
    for t in random.sample(P_miss, 5):
        print(t)
    print()
    print(f'recall miss:')
    for t in random.sample(R_miss, 5):
        print(t)


eval('../train_relations.txt')
print()


def rule_retired(annotations_file):
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        for line in in_f:
            sent_id, person, _, org, sent = line.split('\t')

            tokens = sent.split()
            if 'retired' in tokens:
                p_idx = tokens.index(person.split()[-1])
                o_idx = tokens.index(org.split()[-1])
                r_idx = tokens.index('retired')
                if (p_idx < r_idx < o_idx) or (o_idx < r_idx < p_idx):
                    if r_idx + 2 >= o_idx:
                        print(sent)

print('RETIRED RULE')
rule_retired('../train_relations.txt')

def rule_org_s(annotations_file):
    with open('../data/lexicon.location', 'r', encoding='utf8') as f:
        lex_loc = set([loc.strip() for loc in f])
    with open(annotations_file, 'r', encoding="utf8") as in_f:
        for line in in_f:
            sent_id, person, _, org, sent = line.split('\t')

            tokens = sent.split()
            if "'s" in tokens:
                p_idx = tokens.index(person.split()[-1])
                o_idx = tokens.index(org.split()[-1])
                s_idx = tokens.index("'s")
                if o_idx < s_idx < p_idx:
                    if s_idx - 1 == o_idx:
                        if org in lex_loc:
                            print(sent)

print('ORG S RULE')
rule_org_s('../train_relations.txt')