from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import GATConv
from code.embed import SemAttn
import numpy as np
import os


class HetGAT_Emb(nn.Module, ABC):
    def __init__(self, args, device):
        super(HetGAT_Emb, self).__init__()
        self.device = device
        self.edge_list, self.g_list, self.feat_list = [], [], []
        self.num_ques = args.num_ques
        self.metaPaths = args.meta_paths
        self.fusion = args.fusion
        self.num_metaPath = len(self.metaPaths)

        self.gen_edges_feats(args)

        self.het_gat = HetGAT(self.g_list, self.feat_list, args.emb_dim, self.num_metaPath, args.gat_heads)

        self.semantic_attention = self.get_semantic_attention(args)

        self.element_weights = None
        self.semantic_weight = None

    def forward(self, pad_ques):
        emb_list, self.element_weights = self.het_gat()

        ques_emb_list = [emb[:self.num_ques, :] for emb in emb_list]
        ques_emb, self.semantic_weight = self.semantic_attention(ques_emb_list)  # (N, D)
        batch_ques_emb = F.embedding(pad_ques, ques_emb)  # (L, B, D)
        return batch_ques_emb

    def gen_edges_feats(self, args):
        # generate random embeddings for each type of entity
        entity2emb_dict, entity2num_dict = {}, {}
        type2name_dict = {'q': 'question', 'u': 'train_user', 'k': 'skill', 't': 'template'}
        type_str = '_'.join(args.meta_paths)
        for t in ['q', 'u', 'k', 't']:
            if t in type_str:
                entity2num_dict[t] = len(eval(open(os.path.join(
                    args.data_path, args.data_set, "encode", "%s_id_dict.txt" % type2name_dict[t])).read()))
                entity2emb_dict[t] = nn.Parameter(torch.randn(entity2num_dict[t], args.emb_dim), requires_grad=True)

        # get edges and initial features
        for mp in self.metaPaths:
            feature = torch.cat([entity2emb_dict[mp[0]], entity2emb_dict[mp[1]]], dim=0).to(self.device)
            self.feat_list.append(feature)

            edge = np.load("%s/%s/adj_mat/%s_Edge.npy" % (args.data_path, args.data_set, mp))
            print('num edge of different meta-path, %s: %s' % (mp, edge.shape[1]))
            miss_node_set = set(range(entity2num_dict[mp[0]] + entity2num_dict[mp[1]])) - set(edge.flatten())
            miss_node_np = np.array(list(miss_node_set), dtype=np.int)
            g = dgl.graph((np.concatenate([edge[0], miss_node_np], axis=-1),
                           np.concatenate([edge[1], miss_node_np], axis=-1))).to(self.device)
            self.edge_list.append(torch.stack([g.edges()[0], g.edges()[1]], dim=0))
            self.g_list.append(g)

    def get_semantic_attention(self, args):
        assert self.fusion in ['attnVec_dot', 'attnVec_nonLinear', 'attnVec_topK']
        if self.fusion == 'attnVec_dot':
            return SemAttn.attnVec_dot(args, self.num_metaPath, self.device)
        elif self.fusion == 'attnVec_nonLinear':
            return SemAttn.attnVec_nonLinear(args, self.num_metaPath, self.device)
        else:
            return SemAttn.attnVec_topK(args, self.num_metaPath, self.device)


class HetGAT(nn.Module, ABC):
    def __init__(self, gs, fs, emb_dim, num_path, heads_list):
        super(HetGAT, self).__init__()
        self.num_metaPath = num_path
        self.gs = gs
        self.fs = fs

        # One GAT net for each meta-path
        self.gat_list = nn.ModuleList()
        for i in range(self.num_metaPath):
            self.gat_list.append(GAT(self.gs[i], self.fs[i], emb_dim, len(heads_list[i]), heads_list[i]))

    def forward(self):
        semantic_embeddings, element_weights = [], []
        for i in range(self.num_metaPath):
            emb, wgt = self.gat_list[i]()
            semantic_embeddings.append(emb)
            element_weights.append(wgt)

        return semantic_embeddings, element_weights


class GAT(nn.Module, ABC):
    def __init__(self, g, f, in_dim, num_layers, heads):
        super(GAT, self).__init__()
        self.g = g
        self.f = f
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(GATConv(in_dim, in_dim // heads[i], heads[i], feat_drop=0.2, attn_drop=0.2,
                                           residual=True, activation=F.elu, allow_zero_in_degree=True))

    def forward(self):
        attn_wgt = None
        h = self.f
        for i in range(self.num_layers):
            h, attn_wgt = self.gat_layers[i](self.g, h, get_attention=True)
            h = h.flatten(1)
        return h, attn_wgt
