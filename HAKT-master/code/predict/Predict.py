from abc import ABC
import torch
import torch.nn as nn


class predict_2(nn.Module, ABC):
    def __init__(self, ks_dim, emb_dim, predict_type, num_hidden_layer):
        super(predict_2, self).__init__()
        assert predict_type in ['dot', 'mlp']
        self.predict_type = predict_type

        if self.predict_type == 'mlp':
            MLP_modules = []
            input_size = ks_dim + emb_dim
            for i in range(num_hidden_layer):
                MLP_modules.append(nn.Dropout(p=0.2))
                MLP_modules.append(nn.Linear(input_size, input_size // 2))
                MLP_modules.append(nn.ReLU())
                input_size = input_size // 2
            MLP_modules.append(nn.Dropout(p=0.2))
            MLP_modules.append(nn.Linear(input_size, 1))
            MLP_modules.append(nn.Sigmoid())
            self.MLP_layers = nn.Sequential(*MLP_modules)

    def forward(self, state_emb, next_emb):
        if self.predict_type == 'dot':
            prediction = torch.sigmoid(torch.sum(state_emb * next_emb, dim=-1, keepdim=False))
        else:
            prediction = self.MLP_layers(torch.cat((state_emb, next_emb), -1))
        return prediction.squeeze(-1)
