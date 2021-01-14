# Relation Extration 

This software extract Work_For relation within text.


### Software dependencies:
```bash
git clone https://github.com/shon-otmazgin/nlp_relation_extraction.git
pip install -r requirements.txt
```

### Train:
To train our Relation Extraction model please provide 3 files:
1. ```corpus``` file in format of ```sentid<TAB>sent```
2. ```annotaion``` file in format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB>```
3. ```word_embeddings``` file. We download ```glove.42B.300d.zip``` from the [GloVe web page](https://nlp.stanford.edu/projects/glove/), and extract it into ```data/glove.42B.300d.txt```.

Example:
```python TrainRE data\Corpus.TRAIN.txt data\TRAIN.annotations data/glove.42B.300d.txt```

Program's output are 2 files: ```model``` and ```vectorizer```

