# -*- coding: utf-8 -*-
# Copyright 2019 Huairuo.ai.
# Author: Lin Li (li.lin@huairuo.ai)
#
# cython: language_level=3
#

import os
import torch

import pandas as pd
from my_py_toolkit.file.file_toolkit import writejson

train_path = "./data/train.csv"
test_path = "./data/test.csv"

train_process = "./data/train_process.csv"
test_process = "./data/test_process.csv"

paths = {
  "train": train_path,
  "test": test_path
}

process_path = {
  "train": train_path,
  "test": test_path
}

class HousePrice(torch.nn.Module):
  """
  房价预测。
  """
  def __init__(self, input_size, output_size):
    """

    Args:
      input_size:  82876 维特征,删除了 Summary, Sold Price ，对离散值进行 Onehot编码
      output_size:
    """
    self.line = torch.nn.Linear(input_size, output_size)

  def forward(self, input):
    """

    Args:
      input:

    Returns:

    """
    return self.line(input)

def get_loss(predicts, labels):
  """

  Args:
    predicts:
    labels:

  Returns:

  """
  return (labels - predicts) / labels

def get_data(d_type="train", batch_size=1):
  """

  Args:
    d_type:

  Returns:

  """

  data = pd.read_csv(paths.get(d_type)).iloc[:, 1:]
  num_idx = data.dtypes[data.dtypes != 'object'].index
  data[num_idx] = data[num_idx].apply(lambda x: (x - x.mean()) / x.std())
  data[num_idx] = data[num_idx].fillna(0)
  labels = torch.tensor(data["Sold Price"].values, d_type=torch.float32)
  p_path = process_path.get(d_type)
  if os.path.exists(p_path):
    data = pd.read_csv(paths)
    features = torch.tensor(data, d_type=torch.float)
  else:
    data.drop(columns=["Summary", "Sold Price"])
    data = pd.get_dummies(data)
    data.to_csv(p_path)
    features = torch.tensor(data.values, d_type=torch.long)
  dataset = torch.utils.data.TensorDataSet(features, labels)
  data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True)
  return data_loader

def get_test_loss(test_data, net):
  test_loss = []
  for features, label in test_data:
    test_loss.extend(get_loss(net(features), label).items())
  return sum(test_loss)/len(test_loss)

def main():
  lr = 1e-3
  weight_decry = 0
  epochs = 5
  batch_size = 2

  train_loss = []
  test_loss = []
  net = HousePrice()
  params = filter(lambda param: param.requires_grad, net.parameters())
  adm = torch.optim.Adam(params, lr=lr, weight_decry=weight_decry)
  train_data = get_data("train", batch_size)
  test_data = get_data("test", batch_size)


  for epoch in epochs:
    train_loss = []
    for features, labels in train_data:
      adm.zero_grad()
      predict = net(features)
      losses = get_loss(predict, labels)
      losses.backward()
      adm.step()
    print(f"{epoch}_{step}: loss:{losses.items()}")
    torch.save(net.state_dict(), f"./model/net_{epoch}.pkl")
    test_loss = get_test_loss(test_data, net)
    print(f"epoch: {epoch}, trainloss: {sum(train_loss)/len(train_loss)}")




def test():
  data = pd.read_csv(train_path).drop(columns=["Summary", "Sold Price"]).iloc[:, 1:]
  num_idx = data.dtypes[data.dtypes!='object'].index
  data[num_idx] = data[num_idx].apply(lambda x: (x-x.mean())/x.std())
  # 看看是不是这里导致特征数爆炸
  data[num_idx] = data[num_idx].fillna(0)
  # 删除 Summary 最后剩余 82877 列，继续删除 Sold Price 剩余：82876
  res= pd.get_dummies(data).columns.values.tolist()
  writejson(res, "./columns.json")
  # print(data.)

  # data = pd.read_csv(train_path)
  # d1 = pd.get_dummies(data.iloc[:, 1:], dummy_na=True)
  # print(d1)
  # d1 = pd.get_dummies(pd.Series(data.iloc[:, 1:]))

if __name__ == "__main__":
  # pd.get_dummies([1,2,3,4]).columns
  test()