
# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, Sequential,JumpingKnowledge,global_mean_pool
from torch_geometric_temporal.nn.attention.stgcn import STConv
from torch_geometric_temporal.nn.recurrent import GConvGRU

filters = 32


class social_stgcn(torch.nn.Module):
    def __init__(self, input_feat=2, hidden_channels=5, output_feat=5,
                 stgcn_kernel_size=3, num_nodes=1, K=1):
        super(social_stgcn, self).__init__()

        self.input_feat = input_feat
        self.output_feat = output_feat
        self.K = K
        self.num_nodes = 48

        # self.st_gcns = STConv(in_channels=input_feat,
        #                       hidden_channels=hidden_channels,
        #                       out_channels=output_feat,
        #                       kernel_size=stgcn_kernel_size,
        #                       num_nodes=num_nodes,
        #                       K=K)
        self.gcn = Sequential('x, edge_index, batch', [
            (GCNConv(in_channels=self.input_feat,
                     out_channels=self.output_feat,
                     improved=True,
                     normalize=True,
                     bias=True),
             'x, edge_index -> x1'),
            nn.ReLU(inplace=True),
            (GConvGRU(in_channels=self.output_feat,
                     out_channels=self.output_feat,
                     K=self.K,
                     bias=True),
             'x1, edge_index -> x2'),
            nn.ReLU(inplace=True),
            nn.Dropout(inplace=True),
            # (GCNConv(in_channels=output_feat, out_channels=1), 'x1, edge_index -> x2'),
            # nn.ReLU(inplace=True),
            # (lambda x1, x2: [x1, x2], 'x1, x2 -> xs'),
            # (JumpingKnowledge("cat", 64, num_layers=2), 'xs -> x'),
            # (global_mean_pool, 'x, batch -> x'),
            nn.Linear(in_features=self.output_feat, out_features=3),
        ])

        # self.gcn2 = torch.nn.Sequential(GCNConv(in_channels=output_feat,
        #                                          out_channels=1))
        #
        # self.hidden = [GCNConv(in_channels=output_feat,
        #                        out_channels=output_feat)
        #                for _ in range(hidden_channels-1)]

        # self.hidden = torch.nn.ModuleList(self.hidden)
        # self.recurrent = GConvGRU(input_feat, filters, 2)
        # self.linear = torch.nn.Linear(filters, 1)

    def forward(self, data, device):
        x, edge_index, batch, y = data.x.cuda(), \
                                  data.edge_index.cuda(), \
                                  data.batch.cuda(), \
                                  data.y.cuda()

        x = torch.cat([x,
                       torch.zeros(size=(self.input_feat-x.size()[0],
                                         x.size()[1]), device=device)], 0)
        x = torch.cat([x,
                       torch.zeros(size=(x.size()[0],
                                         self.input_feat - x.size()[1]), device=device)], 1)

        return self.gcn(x, edge_index, batch) #F.log_softmax(x, dim=1)