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
    if ent1.end < ent1.start:
        words_bw = set([w.text for w in sent[ent1.end:ent2.start] if not w.is_punct])
    else:
        words_bw = set([w.text for w in sent[ent2.end:ent1.start] if not w.is_punct])
    if words_bw:
        return words_bw
    else:
        return set(['<EMPTY>'])


def extract_features(ent1, ent2, sent):
    features = {}

    # WORDS FEATURES
    # features['ent1_head'] = ent1.root.text
    # features['ent2_head'] = ent2.root.text
    # features['ent1_ent2_head'] = ent1.root.text + " " + ent2.root.text

    features['bow_ent1_ent2'] = set([w.text for w in ent1] + [w.text for w in ent2])

    features['before_ent1'], features['after_ent1'] = get_before_after(ent1, sent)
    features['before_ent2'], features['after_ent2'] = get_before_after(ent2, sent)

    features['words_between'] = words_between(ent1, ent2, sent)

    # features['ent1_pos'] = ent1.root.tag_
    # features['ent2_pos'] = ent2.root.tag_

    # NER FEATURES
    # features['ent1_type'] = ent1.root.ent_type_
    # features['ent2_type'] = ent2.root.ent_type_
    # features['ent1_ent2_type'] = ent1.root.ent_type_ + " " + ent2.root.ent_type_
    # features['dep_path'] = dependency_path(ent1, ent2)

    return features


def dependency_path(ent1, ent2):
    # print(*[
    #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
    #     for sent in doc.sentences for word in sent.words], sep='\n')

    ent1_path = []
    tok = ent1.root
    while tok.dep_ != 'ROOT':
        ent1_path.append(tok.text)
        ent1_path.append('IN' + "_" + tok.dep_)
        tok = tok.head

    ent2_path = []
    tok = ent2.root
    while True:
        ent2_path.insert(0, tok.text)
        if tok.dep_ == 'ROOT':
            break
        ent2_path.insert(0, 'OUT' + "_" + tok.dep_)
        tok = tok.head

    return ent1_path + ent2_path


def get_y(file, df):
    gold_annotations = read_annotations(file)
    print(f'{WORK_FOR} input annotations: {sum([len(annotations) for annotations in gold_annotations.values()])}')
    y = np.zeros(df.shape[0])

    for i, idx in enumerate(df.index):
        sent_id, person, org, _,  = idx

        for ann in gold_annotations[sent_id]:
            if (person in ann[0] or ann[0] in person) and (org in ann[2] or ann[2] in org):
                y[i] = 1
                break
    return y


def build_df(file, v=None):
    # spa_nlp = spacy.load('en_core_web_lg')
    stanza.download('en')
    # nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse,ner', tokenize_pretokenized=True)

    snlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True)
    nlp = StanzaLanguage(snlp)

    E = []
    F = []
    indices = [[], [], [], []]
    for sent_id, sent_str in tqdm(read_lines(file)):
        # if sent_id == 'sent420':
        #     print('aaa')
        # spa_sent = spa_nlp(sent_str)
        sent = nlp(sent_str)
        persons = [ent for ent in sent.ents if ent.label_ == 'PERSON']
        orgs = [ent for ent in sent.ents if ent.label_ == 'ORG']
        for p, o in itertools.product(persons, orgs):
            features = extract_features(p, o, sent)
            F.append(features)

            # embedding = np.hstack([p.vector.copy(), o.vector.copy()])
            # E.append(embedding)

            indices[0].append(sent_id)
            indices[1].append(p.text)
            indices[2].append(o.text)
            indices[3].append(f'( {sent.text} )')
    if v:
        X = v.transform(F)
    else:
        v = DictVectorizer(sparse=True)
        X = v.fit_transform(F)

    df = pd.concat([pd.DataFrame(E), pd.DataFrame(X.toarray(), columns=v.feature_names_)], axis=1)
    df.index = pd.MultiIndex.from_arrays(indices, names=('sent_id', 'person', 'org', 'sent'))

    return df, v


train_df, v = build_df(file='data/Corpus.TRAIN.txt')
train_y = get_y(file='data/TRAIN.annotations', df=train_df)
print(f'Train size: {train_df.shape}, y: {train_y.shape}, y=1: {train_y[train_y==1].shape}')

dev_df, v = build_df(file='data/Corpus.DEV.txt', v=v)
dev_y = get_y(file='data/DEV.annotations', df=dev_df)
print(f'Dev size: {dev_df.shape}, y: {dev_y.shape}, y=1: {dev_y[dev_y==1].shape}')

with open('pickles/train', 'wb') as f:
    pickle.dump((train_df, train_y), f, pickle.HIGHEST_PROTOCOL)
with open('pickles/dev', 'wb') as f:
    pickle.dump((dev_df, dev_y), f, pickle.HIGHEST_PROTOCOL)

with open('data/TRAIN.annotations_work_for_pred', 'w', encoding="utf8") as f:
    train_df['y'] = train_y
    for idx in train_df[train_df['y']==1].index:
        f.write(f'{idx[0]}\t{idx[1]}\tWork_For\t{idx[2]}\n')
