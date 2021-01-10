from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)

movie_entry = [{'category': ['thriller', 'drama'], 'year': 2003},
               {'category': ['animation', 'family'], 'year': 2011},
               {'year': 1974}]
print(vec.fit_transform(movie_entry))

print(vec.get_feature_names())

print(vec.transform({'category': ['thriller'], 'unseen_feature': '3'}).toarray())