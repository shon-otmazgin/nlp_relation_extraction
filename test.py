# from sklearn.feature_extraction import DictVectorizer
# vec = DictVectorizer(sparse=False)
#
# movie_entry = [{'category': ['thriller', 'drama'], 'year': 2003},
#                {'category': ['animation', 'family'], 'year': 2011},
#                {'year': 1974}]
# print(vec.fit_transform(movie_entry))
#
# print(vec.get_feature_names())
#
# print(vec.transform({'category': ['thriller'], 'unseen_feature': '3'}).toarray())

import spacy
from benepar.spacy_plugin import BeneparComponent
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(BeneparComponent("benepar_en2"))
sent = nlp("American Airlines , a unit of AMR , immediately matched the move , spokesman Tim Wagner said .")
sent1 = list(sent.sents)[0]
for p in list(sent1._.constituents):
    print(p)
    print(p._.labels)