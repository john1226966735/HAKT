from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

from code.embed.HetGAT_Emb import HetGAT_Emb
from code.state.SAKT import SAKT
from code.predict.Predict import predict_2


class KT_Model(nn.Module, ABC):
    def __init__(self, args, device):
        super(KT_Model, self).__init__()

        self.QuesEmb_Layer = HetGAT_Emb(args, device)

        self.Fusion_Layer = Fusion_Module(args.emb_dim, device)

        self.KS_Layer = SAKT(args, device)

        self.Predict_Layer = predict_2(args.hidden_dim, args.emb_dim, args.predict_type, args.num_hidden_layer)

        self.ques_emb = None
        self.ks_emb = None
        self.attn_weight = None

    def forward(self, pad_ques, pad_answer, pad_next):
        self.ques_emb = self.QuesEmb_Layer(pad_ques)
        next_emb = self.QuesEmb_Layer(pad_next)

        input_emb = self.Fusion_Layer(self.ques_emb, pad_answer)

        self.ks_emb = self.KS_Layer(next_emb, self.ques_emb, input_emb)

        self.attn_weight = self.KS_Layer.attn_layers[-1].prob_attn.float()  # [bs,head,seq,seq]

        pad_predict = self.Predict_Layer(self.ks_emb, next_emb)
        return pad_predict

    def explain_loss(self, pad_answer, pad_predict):
        self.attn_weight = self.KS_Layer.attn_layers[-1].prob_attn  # [bs,head,seq1,seq2]
        self.attn_weight = torch.mean(self.attn_weight, dim=1, keepdim=False).transpose(0, 1).transpose(1, 2)  # [seq1, seq2, bs]
        pad_scores = torch.sum(self.attn_weight * pad_answer.unsqueeze(0), dim=1, keepdim=False)  # [L, B]

        return torch.mean((pad_scores - pad_predict) ** 2) ** 0.5
        # return torch.mean(torch.abs(pad_scores - pad_predict))

    def get_hist_score(self, pad_answer):
        self.attn_weight = self.KS_Layer.attn_layers[-1].prob_attn  # [bs,head,seq1,seq2]
        self.attn_weight = torch.mean(self.attn_weight, dim=1, keepdim=False).transpose(0, 1).transpose(1, 2)  # [seq1, seq2, bs]
        pad_scores = torch.sum(self.attn_weight * pad_answer.unsqueeze(0), dim=1, keepdim=False)  # [L, B]
        return pad_scores


class Fusion_Module(nn.Module, ABC):
    def __init__(self, emb_dim, device):
        super(Fusion_Module, self).__init__()
        self.transform_matrix = torch.zeros(2, emb_dim * 2, device=device)
        self.transform_matrix[0][emb_dim:] = 1.0
        self.transform_matrix[1][:emb_dim] = 1.0

    def forward(self, ques_emb, pad_answer):
        ques_emb = torch.cat((ques_emb, ques_emb), -1)
        answer_emb = F.embedding(pad_answer, self.transform_matrix)
        input_emb = ques_emb * answer_emb
        return input_emb
