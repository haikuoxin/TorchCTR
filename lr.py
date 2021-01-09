''' 
@Author: haikuoxin  
@Date: 2021-01-07 22:44:16  
@Last Modified by:   haikuoxin  
@Last Modified time: 2021-01-07 22:44:16
'''
import torch
import pandas as pd

from inputs import SparseFeat, DenseFeat, build_input_features

data = pd.read_csv('./data/criteo_sample.txt')
print(data.head())

sparse_feat_names = ['C' + str(i) for i in range(1, 27)]
dense_feat_names = ['I' + str(i) for i in range(1, 14)]

label = 'label'
embedding_dim = 10

data[sparse_feat_names] = data[sparse_feat_names].fillna('-1')
data[dense_feat_names] = data[dense_feat_names].fillna(0)

feature_names = sparse_feat_names + dense_feat_names
sparse_columns = [
                SparseFeat(
                    name=sparse_feature, 
                    vocabulary_size=data[sparse_feature].nunique(), 
                    embedding_dim=embedding_dim,
                    use_hash=False,
                    dtype='int32',
                    group_name=sparse_feature) for sparse_feature in sparse_features]

dense_columns = [
                    DenseFeat(name=dense_feature,
                    dimension=1,
                    dtype='float32') for dense_feature in dense_features]

feature_columns = sparse_columns + dense_columns
feature_indexs = build_input_features(feature_columns=feature_columns)


class Linear(torch.nn.Module):
    def __init__(self, sparse_columns, dense_columns, feature_indexs):
        self.sparse_columns = sparse_columns
        self.dense_columns = dense_columns
        self.feature_indexs = feature_indexs
        self.init_std = 0.0001

        self.embedding_dict = torch.nn.ModuleDict(
            {
                sparse_column.name: torch.nn.Embedding(
                    num_embeddings = sparse_column.vocabulary_size,
                    embedding_dim = sparse_column.embedding_dim,
                    padding_idx = None
                )
            } for sparse_column in sparse_columns
        )
        self.fc = torch.nn.Linear(len(self.dense_columns), 1)
        torch.nn.normal_(self.fc.weight, mean=0, std=init_std)

    def forward(self, x):
        sparse_embedding_list = [self.embedding_dict[x[self.feature_indexs[sparse_column.name]]] for sparse_column in sparse_columns]
        dense_value_list = [x[self.feature_indexs[dense_column.name]] for dense_column in dense_columns]
