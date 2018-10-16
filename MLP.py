import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self,):
        super(MLP, self,).__init__()


    def creat2(self,state_n,action_1,action_2):
        self.fc1 = nn.Linear(state_n, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out1 = nn.Linear(50, action_1)
        self.out1.weight.data.normal_(0, 0.1)
        self.out2 = nn.Linear(50, action_2)
        self.out2.weight.data.normal_(0, 0.1)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        action1_value = self.out1(x)
        action2_value = self.out2(x)
        return action1_value, action2_value

