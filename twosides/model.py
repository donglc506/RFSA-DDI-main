from audioop import bias
from bisect import bisect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, GCNConv, SAGEConv, GATConv, ChebConv, GraphConv, global_mean_pool
from torch_geometric.nn.conv import GraphConv
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from layers import CoAttentionLayer, FeatIntegrationLayer

class GlobalAttentionPool(nn.Module):

    def __init__(self, in_channels):
        super(GlobalAttentionPool, self).__init__()
        self.in_channels = in_channels
        self.alpha =0.6
        pooling_conv = "GCNConv"
        fusion_conv = "GATConv"
        self.sbtl_layer = self.conv_selection(pooling_conv, in_channels)
        self.fbtl_layer = nn.Linear(in_channels, 1)
        self.fusion = self.conv_selection(fusion_conv, in_channels, conv_type=1)

    def conv_selection(self, conv, in_channels, conv_type=0):
        if (conv_type == 0):
            out_channels = 1
        elif (conv_type == 1):
            out_channels = in_channels
        if (conv == "GCNConv"):
            return GCNConv(in_channels, out_channels)
        elif (conv == "ChebConv"):
            return ChebConv(in_channels, out_channels, 1)
        elif (conv == "SAGEConv"):
            return SAGEConv(in_channels, out_channels)
        elif (conv == "GATConv"):
            return GATConv(in_channels, out_channels, heads=1, concat=True)
        elif (conv == "GraphConv"):
            return GraphConv(in_channels, out_channels)
        else:
            raise ValueError

    def forward(self, x, edge_index, batch):

        score_s = self.sbtl_layer(x, edge_index)

        score_f = self.fbtl_layer(x)

        score = score_s * self.alpha + score_f * (1 - self.alpha)
        x = self.fusion(x, edge_index)
        scores = softmax(score, batch, dim=0)
        gx = global_add_pool(x * scores, batch)

        return gx

class LinearBlock(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.snd_n_feats = 6 * n_feats
        self.lin1 = nn.Sequential(
            nn.BatchNorm1d(n_feats),
            nn.Linear(n_feats, self.snd_n_feats),
        )
        self.lin2 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin3 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats),
        )
        self.lin4 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, self.snd_n_feats)
        )
        self.lin5 = nn.Sequential(
            nn.BatchNorm1d(self.snd_n_feats),
            nn.PReLU(),
            nn.Linear(self.snd_n_feats, n_feats)
        )

    def forward(self, x):
        x = self.lin1(x)
        x = (self.lin3(self.lin2(x)) + x) / 2
        x = (self.lin4(x) + x) / 2
        x = self.lin5(x)

        return x   

class GSP_DMPNN(nn.Module):
    def __init__(self, edge_dim, n_feats, n_iter):
        super().__init__()
        self.n_iter = n_iter

        self.lin_u = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_v = nn.Linear(n_feats, n_feats, bias=False)
        self.lin_edge = nn.Linear(edge_dim, n_feats, bias=False)

        self.att = GlobalAttentionPool(n_feats)
        self.a = nn.Parameter(torch.zeros(1, n_feats, n_iter))
        self.lin_gout = nn.Linear(n_feats, n_feats)
        self.a_bias = nn.Parameter(torch.zeros(1, 1, n_iter))

        glorot(self.a)

        self.lin_block = LinearBlock(n_feats)

    def forward(self, data):

        edge_index = data.edge_index

        edge_u = self.lin_u(data.x)
        edge_v = self.lin_v(data.x)
        edge_uv = self.lin_edge(data.edge_attr)
        edge_attr = (edge_u[edge_index[0]] + edge_v[edge_index[1]] + edge_uv) / 3
        out = edge_attr
        
        out_list = []
        gout_list = []
        for n in range(self.n_iter):
            out = scatter(out[data.line_graph_edge_index[0]] , data.line_graph_edge_index[1], dim_size=edge_attr.size(0), dim=0, reduce='add')
            out = edge_attr + out
            gout = self.att(out, data.line_graph_edge_index, data.edge_index_batch)
            out_list.append(out)
            gout_list.append(F.tanh((self.lin_gout(gout))))

        gout_all = torch.stack(gout_list, dim=-1)
        out_all = torch.stack(out_list, dim=-1)
        scores = (gout_all * self.a).sum(1, keepdim=True) + self.a_bias
        scores = torch.softmax(scores, dim=-1)
        scores = scores.repeat_interleave(degree(data.edge_index_batch, dtype=data.edge_index_batch.dtype), dim=0)

        out = (out_all * scores).sum(-1)

        x = data.x + scatter(out , edge_index[1], dim_size=data.x.size(0), dim=0, reduce='add')
        x = self.lin_block(x)

        return x

class DrugEncoder(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=64, n_iter=10):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), 
        )
        self.lin0 = nn.Linear(in_dim, hidden_dim)
        self.line_graph = GSP_DMPNN(edge_in_dim, hidden_dim, n_iter)

    def forward(self, data):
        data.x = self.mlp(data.x)
        x = self.line_graph(data)

        return x

class RFSA_DDI(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, rel_total, dropout, hidden_dim, n_iter):
        super(RFSA_DDI, self).__init__()
        self.rel_total = rel_total
        self.hidden_dim = hidden_dim
        self.drug_encoder = DrugEncoder(in_dim, edge_in_dim, hidden_dim, n_iter=n_iter)
        self.h_gpool = GlobalAttentionPool(hidden_dim)
        self.t_gpool = GlobalAttentionPool(hidden_dim)
        self.FeatIntegration = FeatIntegrationLayer(hidden_dim, self.rel_total, dropout)
        self.co_attention = CoAttentionLayer(self.hidden_dim)
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.PReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, triples):
        h_data, t_data, rels = triples

        x_h = self.drug_encoder(h_data)
        x_t = self.drug_encoder(t_data)
        h_data.x = x_h
        t_data.x = x_t

        g_h = self.h_gpool(x_h, h_data.edge_index, h_data.batch)
        g_t = self.t_gpool(x_t, t_data.edge_index, t_data.batch)

        g_h, g_t, rels = self.FeatIntegration(g_h, g_t, rels)

        alpha_h, alpha_t = self.co_attention(h_data, t_data, g_h, g_t)

        h_final = scatter(h_data.x * alpha_h.unsqueeze(-1), h_data.batch, reduce='add', dim=0)
        t_final = scatter(t_data.x * alpha_t.unsqueeze(-1), t_data.batch, reduce='add', dim=0)

        scores = self.compute_score(h_final, t_final, rels)

        return scores

    def compute_score(self, h_final, t_final, rels):

        pair_repr = torch.cat([h_final, t_final], dim=-1)

        scores = (self.lin(pair_repr) * rels).sum(-1)
        return scores