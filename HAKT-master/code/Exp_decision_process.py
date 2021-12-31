import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
import os
from code.model import KT_Model

from code.vis_utils import VisTools


def visualize(loader, args):
    visualizer = VisTools(loader, args)

    # visualizer.attention_correlation()
    visualizer.case_study()


def Reliability(loader, args, k):
    device = torch.device(args.device)
    model = KT_Model(args, device).to(device)
    param_file = "params_ASSIST09_128_2021-11-12&01-45-24_0.7907.pkl"
    state_dict = torch.load("../param/%s/%s" % (args.data_set, param_file))
    model.load_state_dict(state_dict)
    model.eval()
    print("load model done!")

    path = 'C:/Users/zjp/OneDrive - mails.ccnu.edu.cn/CODE/KlgTrc/DataProcess2.0/%s/adj_mat' % args.data_set
    # ques_skill_mat = np.load(os.path.join(args.data_path, args.data_set, "graph", "ques_skill_mat.npy"))
    ques_skill_mat = np.load(os.path.join(path, "qk_adj.npy"))
    ques_skill_mat = torch.from_numpy(ques_skill_mat).float().to(device)
    ques_template_mat = np.load(os.path.join(path, "qt_adj.npy"))
    ques_template_mat = torch.from_numpy(ques_template_mat).float().to(device)

    result_list = []
    for seq_lens, pad_data, pad_answer, pad_index, pad_label in loader['test']:
        _ = model(pad_data, pad_answer, pad_index)  # 运行模型

        seq_ques_skill = F.embedding(pad_data, ques_skill_mat)  # [seq, batch, num_skill]
        seq_ques_skill = seq_ques_skill.transpose(0, 1)  # [batch, seq, num_skill]
        seq_ques_template = F.embedding(pad_data, ques_template_mat)  # [seq, batch, num_template]
        seq_ques_template = seq_ques_template.transpose(0, 1)  # [batch, seq, num_template]
        seq_ques_relate_skill = torch.matmul(seq_ques_skill, seq_ques_skill.transpose(1, 2))  # [batch, seq, seq]
        seq_ques_relate_template = torch.matmul(seq_ques_template, seq_ques_template.transpose(1, 2))  # [batch, seq, seq]
        seq_ques_relate = ((seq_ques_relate_skill + seq_ques_relate_template) > 0).int()

        attn_wgt = torch.mean(model.attn_weight, 1, False)  # [batch, seq, seq]

        values, _ = torch.topk(attn_wgt, k=k, dim=2)
        tmp = values[:, :, -1].unsqueeze(-1).repeat(1, 1, attn_wgt.size()[2])
        seq_topk_wgt = (attn_wgt >= tmp).int()  # [batch, seq, seq]

        # # get random attention weight
        # seq_rand_wgt = torch.zeros(size=seq_topk_wgt.size()).to(device)
        # for b in range(seq_topk_wgt.size()[0]):
        #     for i in range(seq_topk_wgt.size()[1]):
        #         for j in np.random.choice(i + 1, k):
        #             seq_rand_wgt[b][i][j] = 1
        # seq_topk_wgt = seq_rand_wgt

        pad_relate = torch.sum(seq_ques_relate * seq_topk_wgt, dim=2)  # [batch, seq]
        pad_relate = (pad_relate > 0).int()
        pad_relate = pad_relate.transpose(0, 1)  # [seq, batch]
        pack_relate = pack_padded_sequence(pad_relate, seq_lens, enforce_sorted=True)
        result_list.append(pack_relate.data.cpu().contiguous().view(-1).detach())

    all_relate = np.concatenate(result_list, 0)
    hit_ratio = np.sum(all_relate) / len(all_relate)
    model.train()
    print(hit_ratio)
    return hit_ratio
