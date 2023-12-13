import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import degree, softmax
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool

class FeatIntegrationLayer(nn.Module):
    def __init__(self, hid_feats, rel_total, dropout=0):
        super().__init__()
        self.hid_feats = hid_feats
        self.rel_total = rel_total
        self.rel_emb = nn.Embedding(self.rel_total, self.hid_feats)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        self.mlp_res = nn.Sequential(
            nn.BatchNorm1d(self.hid_feats * 3),
            nn.Linear(self.hid_feats * 3, self.hid_feats),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )
        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(self.hid_feats * 3),
            nn.Linear(self.hid_feats * 3, self.hid_feats),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )
        self.mlp_tail = nn.Sequential(
            nn.BatchNorm1d(self.hid_feats * 3),
            nn.Linear(self.hid_feats * 3, self.hid_feats),
            nn.PReLU(),
            nn.Dropout(p=dropout)
        )
        self.transGraph = nn.Linear(self.hid_feats, self.hid_feats)


    def forward(self, heads, tails, rels):
        rels_ori = self.rel_emb(rels)

        input = torch.cat([self.transGraph(heads), self.transGraph(tails), rels_ori.squeeze(1)], dim=-1)

        new_rel = self.mlp_res(input)
        rels_new = rels_ori + new_rel


        new_t = self.mlp_tail(input)
        new_h = self.mlp_head(input)

        heads_new = heads + new_h
        tails_new = tails + new_t

        return heads_new, tails_new, rels_new


class CoAttentionLayer(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.w_h = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))
        self.w_t = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))
        self.a = nn.Parameter(torch.zeros(1, self.emb_size))
        self.w_gh = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))
        self.w_gt = nn.Parameter(torch.zeros(self.emb_size, self.emb_size))
        self.bias1 = nn.Parameter(torch.zeros(self.emb_size))
        self.bias2 = nn.Parameter(torch.zeros(self.emb_size))

        nn.init.xavier_uniform_(self.w_t)
        nn.init.xavier_uniform_(self.w_h)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.w_gt)
        nn.init.xavier_uniform_(self.w_gh)
        nn.init.xavier_uniform_(self.bias1.view(-1, *self.bias1.shape))
        nn.init.xavier_uniform_(self.bias2.view(-1, *self.bias1.shape))
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.emb_size, 1),
        )


    def forward(self, h_data, t_data, g_h, g_t):

        g_h_interp = g_h.repeat_interleave(degree(t_data.batch, dtype=t_data.batch.dtype), dim=0)
        g_t_interp = g_t.repeat_interleave(degree(h_data.batch, dtype=h_data.batch.dtype), dim=0)

        alpha_h = h_data.x @ self.w_h + g_t_interp @ self.w_gt + self.bias1
        alpha_t = t_data.x @ self.w_t + g_h_interp @ self.w_gh + self.bias2

        alpha_h = self.mlp(alpha_h).view(-1)
        alpha_t = self.mlp(alpha_t).view(-1)

        alpha_h = softmax(alpha_h, h_data.batch, dim=0)
        alpha_t = softmax(alpha_t, t_data.batch, dim=0)

        return alpha_h, alpha_t
