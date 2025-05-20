import os
import random
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

hidden_layer_dropout = 0.2
def set_seed():
  seed = 0
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  
def dummy_network(sequence_length,cuda):
    # Model architecture
    set_seed()
    
    model = torch.nn.Sequential(
        
    torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1, padding='same'),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30, padding='same'),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30, padding='same'),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40, padding='same'),
    torch.nn.ReLU(), 
    torch.nn.Dropout(p=hidden_layer_dropout),

    torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50, padding='same'),
    torch.nn.ReLU(), 
    torch.nn.Dropout(p=hidden_layer_dropout),

    torch.nn.Flatten(),
    )
    if cuda:
      model.cuda()
    return model

class Seq2Point(nn.Module):
  def __init__(self, sequence_length, cuda):
    set_seed()
    super(Seq2Point, self).__init__()
    dummy_model = dummy_network(sequence_length,cuda)
    rand_tensor = torch.randn(1, 1, sequence_length)
    
    if cuda:
      rand_tensor = rand_tensor.to(device='cuda')
    dummy_output = dummy_model(rand_tensor)
    num_of_flattened_neurons = dummy_output.shape[-1]

    ## Now define the actual network

    self.conv1 = torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1, padding='same')
    nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    self.bn1 = nn.BatchNorm1d(30)
    self.conv2 = torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30, padding='same')
    nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    self.bn2 = nn.BatchNorm1d(30)
    self.conv3 = torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30, padding='same')
    nn.init.kaiming_normal_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
    self.conv4 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40, padding='same')
    nn.init.kaiming_normal_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
    self.conv5 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50, padding='same')
    nn.init.kaiming_normal_(self.conv5.weight, mode='fan_in', nonlinearity='relu')
    self.fc1 = torch.nn.Linear(out_features=1024, in_features=num_of_flattened_neurons)
    self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
    self.dropout1 = torch.nn.Dropout(hidden_layer_dropout)
    self.dropout2 = torch.nn.Dropout(hidden_layer_dropout)
    if cuda:
      self.cuda()

  def forward(self, X):
    x = self.conv1(X)
    x = F.relu(x)
    x = F.relu(self.bn1(x))

    x = self.conv2(x)
    x = F.relu(x)
    x = F.relu(self.bn2(x))

    x = self.conv3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = F.relu(x)

    x = self.dropout1(x)

    x = self.conv5(x)
    x = F.relu(x)
    x = self.dropout2(x)

    x = x.reshape(x.size(0), -1)
    
    x = self.fc1(x)
    x = F.relu(x)

    x = self.fc2(x)
    return x
   