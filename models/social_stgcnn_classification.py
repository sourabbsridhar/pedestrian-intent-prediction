
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Sequential
from torch_geometric_temporal.nn.recurrent import GCLSTM, GConvLSTM

filters = 32

class social_stgcn(torch.nn.Module):
    def __init__(self, input_feat=2, linear_input=0 ,  Conv_outputs=[5], LSTM_output=[5], K=1, linear_output=3):
        super(social_stgcn, self).__init__()

        self.input_feat = input_feat
        self.linear_input = linear_input
        self.Conv_outputs = Conv_outputs
        self.LSTM_output = LSTM_output
        self.linear_output = linear_output
        self.K = K

        self.gcn1 = GCNConv(in_channels=self.input_feat, out_channels=self.input_feat, improved=True)
        self.gcn2 = GCNConv(in_channels=self.input_feat, out_channels=self.input_feat, improved=True)

        self.gclstm1 = GCLSTM(in_channels=self.input_feat, out_channels=self.input_feat, K=K, normalization="sym", bias=True)
        self.gclstm2 = GCLSTM(in_channels=self.input_feat, out_channels=self.input_feat, K=K, normalization="sym", bias=True)
        self.gclstm3 = GCLSTM(in_channels=self.input_feat, out_channels=self.input_feat, K=K, normalization="sym", bias=True)

        self.no_lstm = 3

        self.linear = nn.Linear(in_features=(self.linear_input), out_features=self.linear_output)

        self.softmax = nn.LogSoftmax(dim=0)



    def forward(self, data, no_pedestrians, device):
        x, edge_index = data.x.cuda(), data.edge_index.cuda()
        x = torch.cat([x, torch.zeros(size=(self.input_feat-x.size()[0], x.size()[1]), device=device)], 0)
        x = torch.cat([x, torch.zeros(size=(x.size()[0], self.input_feat - x.size()[1]), device=device)], 1)

        h = [None for i in range(self.no_lstm)]
        c = [None for i in range(self.no_lstm)]

        x = self.gcn1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = self.gcn2(x=x, edge_index=edge_index)
        x = F.relu(x)
        h[0], c[0] = self.gclstm1(x, edge_index, H=h[0], C=c[0])
        h[1], c[1] = self.gclstm2(h[0], edge_index, H=h[1], C=c[1])
        h[2], c[2] = self.gclstm3(h[1], edge_index, H=h[2], C=c[2])

        x = F.relu(h[2])

        output = torch.empty([no_pedestrians, 3, 1], device="cuda")
        x = self.linear(x)
        
        for i in range(no_pedestrians):
            output[i] = self.softmax(x[i].reshape([3, 1]))
        x = output
        return x