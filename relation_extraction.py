from utils import SENTENCES, ANNOTATIONS, LIVE_IN, WORK_FOR


def read_sentences(file):
    with open(file, 'r') as f:
        return dict(line.split('\t') for line in f.read().splitlines())


def read_annotation(file):
    with open(file, 'r') as f:
        return dict([line.split('\t')[0], tuple(line.split('\t')[1:-1])] for line in f.read().splitlines())


train_set = {SENTENCES: read_sentences(file='data\Corpus.TRAIN.txt'),
             ANNOTATIONS: read_annotation(file='data\TRAIN.annotations')}

live = 0
work = 0
for sent_id, annot in train_set[ANNOTATIONS].items():
    if annot[1] == LIVE_IN: live += 1
    elif annot[1] == WORK_FOR: work += 1

print(live)
print(work)
