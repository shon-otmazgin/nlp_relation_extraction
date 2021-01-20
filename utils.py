import codecs
from collections import defaultdict
import sys

WORK_FOR = 'Work_For'


def read_lines(fname):
    sentences = []
    for line in codecs.open(fname, encoding="utf8"):
        try:
            sent_id, sent = line.strip().split("\t")
        except ValueError:
            print('ERROR: Wrong format file.\n expected format: sentid<TAB>sent')
            sys.exit(-1)
        sent = sent.replace("-LRB-","(")
        sent = sent.replace("-RRB-",")")
        sentences.append((sent_id, sent))
    return sentences


def read_annotations(fname):
    annotations = defaultdict(lambda: [])
    for line in codecs.open(fname, encoding="utf8"):
        try:
            sent_id, arg1, rel, arg2 = line.strip().split("\t")[0:4]
        except ValueError:
            print('ERROR: Wrong format file.\n expected format: sentid<TAB>ent1<TAB>rel<TAB>ent2 OR sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB>( sent )')
            sys.exit(-1)
        if rel != WORK_FOR:
            continue
        annotations[sent_id].append((arg1, rel, arg2))
    return annotations
