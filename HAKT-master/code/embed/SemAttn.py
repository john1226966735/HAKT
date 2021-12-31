from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------- for multiple meta-path fusion ------------------------
class attnVec_dot(nn.Module, ABC):  # train attention vector + dot
    def __init__(self, args, num_mataPath, device):
        super(attnVec_dot, self).__init__()
        self.num_path = num_mataPath
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        # [num_ques, num_metaPath, 1]
        path_weight = F.softmax(torch.matmul(semantic_embeddings, self.attnVec), dim=1)
        # [num_ques, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * path_weight, dim=1, keepdim=False)
        return ques_embedding, path_weight


class attnVec_nonLinear(nn.Module, ABC):  # train attention vector + nonLinear
    def __init__(self, args, num_metaPath, device):
        super(attnVec_nonLinear, self).__init__()
        self.num_path = num_metaPath
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)

        # [num_ques, num_metaPath, 1]
        trans_embeddings = torch.tanh(self.fc(semantic_embeddings))
        path_weight = F.softmax(torch.matmul(trans_embeddings, self.attnVec), dim=1)

        # [num_ques, emb_dim]
        ques_embedding = torch.sum(semantic_embeddings * path_weight, dim=1, keepdim=False)
        return ques_embedding, path_weight


class attnVec_topK(nn.Module, ABC):
    def __init__(self, args, num_metaPath, device):
        super(attnVec_topK, self).__init__()
        self.num_path = num_metaPath
        self.top_k = args.top_k
        self.emb_dim = args.emb_dim
        self.attnVec = nn.Parameter(torch.rand(size=(1, args.emb_dim, 1), device=device), True)
        self.fc = nn.Linear(args.emb_dim, args.emb_dim).to(device)

    def forward(self, semantic_embeddings):
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D)
        # [num_ques, num_metaPath, 1]
        path_weight = torch.matmul(torch.tanh(self.fc(semantic_embeddings)), self.attnVec)
        # path_weight = F.softmax(path_weight, dim=1)

        # topK selection
        select_weight = torch.topk(path_weight, k=self.top_k, dim=1, sorted=False)
        path_weight = select_weight.values
        path_weight = F.softmax(path_weight, dim=1)  # softmax
        index = select_weight.indices.repeat(1, 1, self.emb_dim)
        ques_embeddings = torch.gather(semantic_embeddings, dim=1, index=index)

        # [num_ques, emb_dim]
        ques_embedding = torch.sum(ques_embeddings * path_weight, dim=1, keepdim=False)
        return ques_embedding, path_weight
