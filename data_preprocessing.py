import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("training_set_rel3.tsv", sep='\t', encoding="iso-8859-1")
essays = df.as_matrix(columns=['essay']).ravel()
lengths = []

for essay in essays:
	lengths.append(len(essay.split()))

df = df[['essay_id', 'essay_set', 'essay', 'domain1_score']]
df['essay_length'] = lengths
indices = df.columns.tolist()
indices[2] = 'essay_full'

vec = CountVectorizer(token_pattern='[a-z]{2,}')
X = vec.fit_transform(essays)
count_vec_df = pd.DataFrame(X.todense(), columns=vec.get_feature_names())

totals = count_vec_df.sum(axis=0)
totals = totals[totals >= 30]
print(totals.shape)

count_vec_df = count_vec_df[totals.index.tolist()]
count_vec_df[indices] = df
totals = count_vec_df.sum(axis=0)

tables = []
for i in range(1, 9):
	table = count_vec_df.loc[count_vec_df['essay_set'] == i].drop('essay_set',1)
	table.to_csv('data/nice' + str(i) + '.csv', index=False)
	tables.append(table)