from utils import CORPUS, ANNOTATIONS, PROCESSED_CORPUS, LIVE_IN, WORK_FOR, FIELDS_H


def read_corpus(file):
    with open(file, 'r') as f:
        return dict(line.split('\t') for line in f.read().splitlines())


def read_annotation(file):
    with open(file, 'r') as f:
        return dict((line.split('\t')[0], tuple(line.split('\t')[1:-1])) for line in f.read().splitlines())


def read_processed_corpus(file):
    def tokenize(tokens):
        if len(tokens) == 1:
            return tokens[0]
        return {h: v for h, v in zip(FIELDS_H, tokens)}

    processed_corpus = {}
    with open(file, 'r', encoding='utf8') as f:
        sentence = []
        for line in f:
            line = line.rstrip('\n')
            if line:
                sentence.append(tokenize(line.split('\t')))
            else:
                processed_corpus[sentence[0].split(':')[1].strip()] = sentence[2:]
                sentence = []
    return processed_corpus


train_set = {CORPUS: read_corpus(file='data\Corpus.TRAIN.txt'),
             ANNOTATIONS: read_annotation(file='data\TRAIN.annotations'),
             PROCESSED_CORPUS: read_processed_corpus(file='data\Corpus.TRAIN.processed')}

dev_set = {CORPUS: read_corpus(file='data\Corpus.DEV.txt'),
           ANNOTATIONS: read_annotation(file='data\DEV.annotations'),
           PROCESSED_CORPUS: read_processed_corpus(file='data\Corpus.DEV.processed')}

# live = 0
# work = 0
# for sent_id, annot in train_set[ANNOTATIONS].items():
#     if annot[1] == LIVE_IN: live += 1
#     elif annot[1] == WORK_FOR: work += 1
#
# print(live)
# print(work)
