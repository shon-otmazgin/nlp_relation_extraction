import pandas as pd
import itertools

from spacy.tokens.span import defaultdict

from utils import CORPUS, ANNOTATIONS, PROCESSED_CORPUS, LIVE_IN, WORK_FOR, FIELDS_H, ENTTYPE, FORM, ENTIBO, HEAD, \
    LEMMA, ID, DEPREL
import sys
import numpy as np
import spacy
import codecs
from tqdm import tqdm

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def extract_features(ent1, ent2, sent, sent_id):

    # Entity-based features
    embedding = ent1.root.vector
    embedding += ent2.root.vector

    # Word-based features
    embedding += nlp(sent.text[ent1.end_char:ent2.end_char-len(ent2.text)]).vector
    embedding += nlp(sent.text[:ent1.end_char-len(ent1.text)].split()[-1]).vector if len(sent.text[:ent1.end_char-len(ent1.text)]) > 0 else 0
    embedding += nlp(sent.text[ent2.end_char:].split()[0]).vector if len(sent.text[ent2.end_char:]) > 0 else 0

    # Syntactic features
    # features['constituent_path'] = None
    # features['basic_syntactic_chunk_path'] = None
    # features['typed_dependency_path'] = dependency_path(ent1, ent2)

    features = pd.Series(embedding, name=(sent_id, ent1.text, ent2.text))
    features['ent1_type'] = ent1.root.ent_type_
    features['ent2_type'] = ent2.root.ent_type_
    features['concat_type'] = features['ent1_type'] + features['ent2_type']
    features['y'] = 0
    return features


def dependency_path(ent1, ent2):
    dep_path = []

    tok = ent1.root
    while tok.dep_ != 'ROOT':
        dep_path.append((tok.text, tok.dep_))
        tok = tok.head

    i = len(dep_path)

    tok = ent2.root
    while True:
        dep_path.insert(i, (tok.text, tok.dep_))
        if tok.dep_ == 'ROOT':
            break
        tok = tok.head

    return dep_path


def read_lines(fname):
    sentences = []
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, sent = line.strip().split("\t")
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        sentences.append((sent_id, sent))
    return sentences


def read_annotations(fname):
    annotations = defaultdict(lambda: [])
    for line in codecs.open(fname, encoding="utf8"):
        sent_id, arg1, rel, arg2 = line.strip().split("\t")[0:4]
        annotations[sent_id].append((arg1, rel, arg2))
    return annotations


nlp = spacy.load('en_core_web_lg')

ENTITIES_TYPE = ['PERSON', 'ORG']

train_annotations = read_annotations(sys.argv[2])

c = 0
for sent_id, annotations in train_annotations.items():
    for ann in annotations:
        if ann[1] == WORK_FOR:
            # print(f'{sent_id} {ann}')
            c+=1
print(f'train {WORK_FOR} annotations: {c}')

train_df = pd.DataFrame()
for sent_id, sent_str in tqdm(read_lines(sys.argv[1])):
    sent = nlp(sent_str)
    for ent1, ent2 in itertools.combinations(sent.ents, 2):
        if ent1.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent2.root.ent_type_ not in ENTITIES_TYPE:
            continue
        if ent1.root.ent_type_ == ent2.root.ent_type_:
            continue
        features = extract_features(ent1, ent2, sent, sent_id)
        train_df = train_df.append(features)

train_df = pd.get_dummies(train_df, columns=['concat_type', 'ent1_type', 'ent2_type'])

print(train_df.shape)

