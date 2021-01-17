''' 
@Author: haikuoxin  
@Date: 2021-01-07 22:44:16  
@Last Modified by:   haikuoxin  
@Last Modified time: 2021-01-07 22:44:16
'''
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import roc_auc_score

from inputs import SparseFeat, DenseFeat, build_input_features

data = pd.read_csv('./data/criteo_sample.txt')
sparse_feat_names = ['C' + str(i) for i in range(1, 27)]
dense_feat_names = ['I' + str(i) for i in range(1, 14)]

label = 'label'
embedding_dim = 10

data[sparse_feat_names] = data[sparse_feat_names].fillna('-1')
data[dense_feat_names] = data[dense_feat_names].fillna(0)

for sparse_feat_name in sparse_feat_names:
    lbe = LabelEncoder()
    data[sparse_feat_name] = lbe.fit_transform(data[sparse_feat_name])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_feat_names] = mms.fit_transform(data[dense_feat_names])

feature_names = sparse_feat_names + dense_feat_names
sparse_columns = [
                SparseFeat(
                    name=sparse_feature, 
                    vocabulary_size=data[sparse_feature].nunique() + 1, 
                    embedding_dim=embedding_dim,
                    use_hash=False,
                    dtype='int32',
                    group_name=sparse_feature) for sparse_feature in sparse_feat_names]

dense_columns = [
                    DenseFeat(name=dense_feature,
                    dimension=1,
                    dtype='float32') for dense_feature in dense_feat_names]

feature_columns = sparse_columns + dense_columns
feature_indexs = build_input_features(feature_columns=feature_columns)


class Linear(torch.nn.Module):
    def __init__(self, sparse_columns, dense_columns, feature_indexs):
        super(Linear, self).__init__()
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
                ) for sparse_column in sparse_columns
            }
        )
        self.fc = torch.nn.Linear(len(self.dense_columns), 1)
        # torch.nn.normal_(self.fc.weight, mean=0, std=init_std)

    def forward(self, x):
        sparse_embedding_list = [
            self.embedding_dict[sparse_column.name](
                x[:, self.feature_indexs[sparse_column.name][0]: self.feature_indexs[sparse_column.name][1]].long()
            ) for sparse_column in self.sparse_columns
        ]

        dense_value_list = [
            x[:, self.feature_indexs[dense_column.name][0]: self.feature_indexs[dense_column.name][1]] 
            for dense_column in dense_columns
        ]

        linear_sparse_logit = torch.sum(torch.cat(sparse_embedding_list, dim=-1), dim=-1, keepdim=False)
        linear_dense_logit = self.fc(torch.cat(dense_value_list, dim=-1))
        logit = linear_sparse_logit + linear_dense_logit
        return torch.sigmoid(logit)


if __name__ == "__main__":
    line = 150
    model = Linear(sparse_columns, dense_columns, feature_indexs)
    x = torch.tensor([data.loc[:, col].values for col in feature_indexs.keys()], dtype=torch.float32).T
    y = torch.tensor(data.loc[:, label].values, dtype=torch.float32).T
    x_train = x[:line, :]
    x_test = x[line:, :]
    y_train = y[: line]
    y_test = y[line: ]
    optim = torch.optim.Adam(params=model.parameters(),  lr=0.01)

    for i in range(10):
        model.train()
        output = model(x_train).squeeze()
        optim.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy(output, y_train, reduction='mean')
        auc = roc_auc_score(y_train.long().tolist(), output.tolist())
        loss.backward()
        optim.step()

        with torch.no_grad():
            model.eval()
            output = model(x_test).squeeze()
            loss_test = torch.nn.functional.binary_cross_entropy(output, y_test, reduction='mean')
            auc_test = roc_auc_score(y_test.long().tolist(), output.tolist())
            print (f'{i} \t train loss: {round(loss.item(), 4)} \t train auc: {round(auc, 4)} \t test loss: {round(loss_test.item(), 4)} \t test auc: {round(auc_test, 4)}')
