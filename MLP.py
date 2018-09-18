import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self,):
        super(MLP, self,).__init__()


    def creat2(self,state_n,action_n):
        self.fc1 = nn.Linear(state_n, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, action_n)
        self.out.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value
