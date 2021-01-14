# Relation Extration 

This software extract Work_For relation within text.


### Software dependencies:
```bash
git clone https://github.com/shon-otmazgin/nlp_relation_extraction.git
pip install -r requirements.txt
```

### Train:
To train our Relation Extraction model, please run ```TrainRE.py``` with 3 files:
1. ```corpus``` file in format of ```sentid<TAB>sent```
2. ```annotaion``` file in format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB>```
3. ```word_embeddings``` file. We download ```glove.42B.300d.zip``` from the [GloVe web page](https://nlp.stanford.edu/projects/glove/), and extract it into ```data/glove.42B.300d.txt```.

Example:
```python TrainRE.py data/Corpus.TRAIN.txt data/TRAIN.annotations data/glove.42B.300d.txt```

Program's output are 2 files: ```model``` and ```vectorizer```

### Extract Relations (Inference)
To get relations from trained model, please run ```ExtractRE.py``` with 2 files:
1. ```model``` (produced in the Train phase)
2. ```corpus``` file in format of ```sentid<TAB>sent```

Example:
```python ExtractRE.py model data/Corpus.DEV.txt```

Program's output is a text file named ```predicted_relation``` in the format of: ```sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB> ( sent )```

### Evaluation
To evaluate the result with gold annotations please run ```eval.py``` with 2 files:
1. gold annotations in the format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB>``` 
2. predicted annotations in the format of ```sentid<TAB>ent1<TAB>rel<TAB>ent2<TAB>```

Example:
```python eval.py data/DEV.annotations predicted_relation```

Program's output is the precision, recall and f1 scores.

