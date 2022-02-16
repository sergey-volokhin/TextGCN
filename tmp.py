import pandas as pd
import numpy as np
import dgl


path = 'data/dummy/'
train_df = pd.read_table(path + 'train.tsv', dtype=np.int32, header=0, names=['user_id', 'asin'])
n_users = train_df.user_id.nunique()
n_items = train_df.asin.nunique()

graph = dgl.heterograph({('user', 'bought', 'item'): (train_df['user_id'].values, train_df['asin'].values),
                         ('item', 'bought_by', 'user'): (train_df['asin'].values, train_df['user_id'].values)})

x = graph.adj(etype='bought').coalesce()
print(x.to_dense())
print(x.indices()[1])
test = x.indices()[1] + 5
print(test)
