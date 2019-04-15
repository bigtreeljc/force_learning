import unittest
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
from pointer_network.PointerNet import PointerNet
from pointer_network.Data_Generator import TSPDataset

def get_opt():
  opt = edict()
  opt.train_size=10000
  opt.val_size=1000
  opt.test_size=10000
  opt.batch_size=2
  opt.nof_epoch=50000
  opt.lr=0.0001
  opt.gpu=False
  opt.nof_points=5
  opt.embedding_size=128
  opt.hiddens=512
  opt.nof_lstms=2
  opt.dropout=0.
  opt.bidir=True
  return opt

class test(unittest.TestCase):
  def test_train(self):
    params = get_opt()  
    print(params)
    model = PointerNet(params.embedding_size,
      params.hiddens, params.nof_lstms,
      params.dropout, params.bidir)
    dataset = TSPDataset(params.train_size, params.nof_points)
    dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=4)

    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)

    iterator = iter(dataloader)
    sample_batched = iterator.next()

    train_batch = sample_batched['Points']
    target_batch = sample_batched['Solution']
    print(f"train_batch {train_batch.size()} \ntarget batch {target_batch.size()}")
    print(f"train_batch {train_batch} \ntarget batch {target_batch}")
    train_batch = Variable(sample_batched['Points'])
    target_batch = Variable(sample_batched['Solution'])
    o, p = model(train_batch)
    print(f"o {o}\np {p}\no size {o.size()}\np size {p.size()}")
    o = o.contiguous().view(-1, o.size()[-1])
    print(f"flattened o {o.size()}\n{o}")
    target_batch = target_batch.view(-1)
    loss = CCE(o, target_batch)
    print(loss.item())
    model_optim.zero_grad()
    loss.backward()
    model_optim.step()
    print("parameter updated")


if __name__ == "__main__":
  unittest.main()
