import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("training_set_rel3.tsv", sep='\t', encoding="iso-8859-1")
essays = df.as_matrix(columns=['essay']).ravel()

# Compute lengths of essays, used as a feature
lengths = []
for essay in essays:
	lengths.append(len(essay.split()))

df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]
df['essay_length'] = lengths
indices = df.columns.tolist()
indices[2] = 'essay_full'

# Bigrams
vec = CountVectorizer(token_pattern='[a-z]{2,}', ngram_range=(2,2), min_df=50)
X_bigram = vec.fit_transform(essays)
count_bi_vec_df = pd.DataFrame(X_bigram.todense(), columns=vec.get_feature_names())
print(count_bi_vec_df.shape)
totals = count_bi_vec_df.sum(axis=0)
print(totals.shape)
indices_bigram = count_bi_vec_df.columns.tolist()

# Single grams
vec = CountVectorizer(token_pattern='[a-z]{2,}', min_df=30)
X = vec.fit_transform(essays)
count_vec_df = pd.DataFrame(X.todense(), columns=vec.get_feature_names())
totals = count_vec_df.sum(axis=0)

# Combine all dataframes
count_vec_df = count_vec_df[totals.index.tolist()]
count_vec_df[indices] = df
count_vec_df[indices_bigram] = count_bi_vec_df
totals = count_vec_df.sum(axis=0)

tables = []
for i in range(1, 9):
	table = count_vec_df.loc[count_vec_df['essay_set'] == i].drop('essay_set',1)
	table.to_csv('data/nice' + str(i) + '.csv', index=False)
	tables.append(table)