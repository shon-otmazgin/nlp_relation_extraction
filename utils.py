import codecs
from collections import defaultdict

WORK_FOR = 'Work_For'
ENTITIES_TYPE = ['PERSON', 'ORG']
stop_words = ['the', "'s"]

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
    with open(fname+"_work_for", 'w', encoding="utf8") as f:
        for line in codecs.open(fname, encoding="utf8"):
            sent_id, arg1, rel, arg2 = line.strip().split("\t")[0:4]
            if rel != WORK_FOR:
                continue
            annotations[sent_id].append((arg1, rel, arg2))
            f.write(f'{sent_id}\t{arg1}\tWork_For\t{arg2}\n')
    return annotations
