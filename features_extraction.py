import pandas as pd
import itertools
from utils import read_lines, ENTITIES_TYPE, stop_words, WORK_FOR, read_annotations
import sys
import spacy
import numpy as np
import pickle
from sklearn.feature_extraction import DictVectorizer
import stanza
from spacy_stanza import StanzaLanguage
import string
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from spacy.vocab import Vocab
stanza.download('en')

pd.set_option("display.max_rows", None, "display.max_columns", None, 'display.width', None)


def get_before_after(ent, sent):
    before = '<START>'
    for i in range(1, len(sent)):
        if ent.start - i > 0 and not sent[ent.start - i].is_punct:
            before = sent[ent.start - i].text
            break

    after = '<END>'
    for i in range(len(sent)):
        if ent.end + i < len(sent) and not sent[ent.end + i].is_punct:
            after = sent[ent.end + i].text
            break
    return before, after


def words_between(ent1, ent2, sent):
    if ent1.start > ent2.start:
        words_bw = sent[ent2.end:ent1.start]
    else:
        words_bw = sent[ent1.end:ent2.start]
    if words_bw:
        return [w.text for w in words_bw if not w.is_punct]
    else:
        return ['<EMPTY>']


def extract_features(ent1, ent2, sent):
    features = {}

    features['bow_ent1_ent2'] = [w.text for w in ent1] + [w.text for w in ent2]
    features['before_ent1'], features['after_ent1'] = get_before_after(ent1, sent)
    features['before_ent2'], features['after_ent2'] = get_before_after(ent2, sent)
    features['words_between'] = words_between(ent1, ent2, sent)

    features['dep_path'] = dependency_path(ent1, ent2)

    return features


def dependency_path(ent1, ent2):
    ent1_path, ent2_path = [], []
    tok1, tok2 = ent1.root, ent2.root
    while tok1.dep_ != 'root' or tok2.dep_ != 'root':
        if tok1.dep_ != 'root':
            ent1_path.append(tok1.text)
            ent1_path.append(tok1.dep_)
            tok1 = tok1.head
        if tok2.dep_ != 'root':
            ent2_path.append(tok2.text)
            ent2_path.append(tok2.dep_)
            tok2 = tok2.head
        if tok1 == tok2:
            break
    return ent1_path + [tok1.text] + ent2_path[::-1] if ent1.start < ent2.start else ent2_path + [tok1.text] + ent1_path[::-1]


def get_y(file, df):
    gold_annotations = read_annotations(file)
    print(f'{WORK_FOR} input annotations: {sum([len(annotations) for annotations in gold_annotations.values()])}')
    y = np.zeros(df.shape[0])

    for i, idx in enumerate(df.index):
        sent_id, person, org, _, = idx

        for ann in gold_annotations[sent_id]:
            if (person in ann[0] or ann[0] in person) and (org in ann[2] or ann[2] in org):
                y[i] = 1
                break
    return y


def features2vectors(F, V):
    ent1_ent2_bow = [' '.join(f['bow_ent1_ent2']) for f in F]
    before_ent1 = [f['before_ent1'] for f in F]
    after_ent2 = [f['after_ent2'] for f in F]
    wb_bow = [' '.join(f['words_between']) for f in F]
    dep_path_bow = [' '.join(f['dep_path']) for f in F]

    if not V:
        ent1_ent2_v = CountVectorizer(analyzer="word", binary=True, ngram_range=(1, 2)).fit(ent1_ent2_bow)
        before_ent1_v = CountVectorizer(analyzer="word", binary=True, ngram_range=(1, 2)).fit(before_ent1)
        after_ent2_v = CountVectorizer(analyzer="word", binary=True, ngram_range=(1, 2)).fit(after_ent2)
        wb_bow_v = CountVectorizer(analyzer="word", binary=True, ngram_range=(1, 2)).fit(wb_bow)
        dep_path_bow_v = CountVectorizer(analyzer="word", binary=True, ngram_range=(2, 4)).fit(dep_path_bow)
    else:
        ent1_ent2_v, before_ent1_v, after_ent2_v, wb_bow_v, dep_path_bow_v = V

    return np.hstack([ent1_ent2_v.transform(ent1_ent2_bow).toarray(),
                      before_ent1_v.transform(before_ent1).toarray(),
                      after_ent2_v.transform(after_ent2).toarray(),
                      wb_bow_v.transform(wb_bow).toarray(),
                      dep_path_bow_v.transform(dep_path_bow).toarray()]), (
           ent1_ent2_v, before_ent1_v, after_ent2_v, wb_bow_v, dep_path_bow_v)


def read_vectors():
    def load_embeddings():
        df = pd.read_pickle('pickles/embeddings_df')
        return zip(list(df.index), np.array(df.values))

    vocab = Vocab()
    for word, vector in load_embeddings():
        vocab.set_vector(word, vector)
    return vocab


def get_vector(span, vocab):
    return np.mean([vocab.get_vector(w.lemma_.lower()) for w in span], axis=0)


def build_df(file, V=None):
    snlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)
    nlp = StanzaLanguage(snlp)
    vocab = read_vectors()

    E = []
    F = []
    indices = [[], [], [], []]
    for sent_id, sent_str in tqdm(read_lines(file)):
        # if sent_id == 'sent95':
        #     print('aaa')
        # sent = nlp(sent_str)
        sent = nlp(sent_str)
        persons = [ent for ent in sent.ents if ent.label_ == 'PERSON']
        orgs = [ent for ent in sent.ents if ent.label_ == 'ORG']
        for p, o in itertools.product(persons, orgs):
            features = extract_features(p, o, sent)
            F.append(features)

            embedding = np.hstack([get_vector(p, vocab), get_vector(o, vocab)])
            E.append(embedding)

            indices[0].append(sent_id)
            indices[1].append(p.text)
            indices[2].append(o.text)
            indices[3].append(f'( {sent.text} )')
    X, V = features2vectors(F, V)

    df = pd.concat([pd.DataFrame(E), pd.DataFrame(X)], axis=1)
    df.index = pd.MultiIndex.from_arrays(indices, names=('sent_id', 'person', 'org', 'sent'))

    return df, V


train_df, V = build_df(file='data/Corpus.TRAIN.txt', V=None)
train_y = get_y(file='data/TRAIN.annotations', df=train_df)
print(f'Train size: {train_df.shape}, y: {train_y.shape}, y=1: {train_y[train_y == 1].shape}')

dev_df, V = build_df(file='data/Corpus.DEV.txt', V=V)
dev_y = get_y(file='data/DEV.annotations', df=dev_df)
print(f'Dev size: {dev_df.shape}, y: {dev_y.shape}, y=1: {dev_y[dev_y == 1].shape}')

with open('pickles/train', 'wb') as f:
    pickle.dump((train_df, train_y), f, pickle.HIGHEST_PROTOCOL)
with open('pickles/dev', 'wb') as f:
    pickle.dump((dev_df, dev_y), f, pickle.HIGHEST_PROTOCOL)