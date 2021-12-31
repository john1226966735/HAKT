import torch
from torch.nn.utils.rnn import pack_padded_sequence
import matplotlib.pyplot as plt
import numpy as np
import os
from code.model import KT_Model


def get_fidelity(loader, args):
    device = torch.device(args.device)
    model = KT_Model(args, device).to(device)

    for file in os.listdir("../param/%s" % args.data_set):
        model.load_state_dict(torch.load("../param/%s/%s" % (args.data_set, file)), strict=False)

        model.eval()
        score_list, pred_list = [], []
        thresh_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        for seq_lens, pad_data, pad_answer, pad_index, pad_label in loader['test']:
            pad_predict = model(pad_data, pad_answer, pad_index)  # 运行模型
            pack_predict = pack_padded_sequence(pad_predict, seq_lens, enforce_sorted=True)
            pad_hist_score = model.get_hist_score(pad_answer)
            pack_hist_score = pack_padded_sequence(pad_hist_score, seq_lens, enforce_sorted=True)

            y_hist_score = pack_hist_score.data.cpu().contiguous().view(-1).detach()
            y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()

            score_list.append(y_hist_score)
            pred_list.append(y_pred)

        fidelity_list = []
        for thresh in thresh_list:
            hist_score = np.concatenate(score_list, 0)
            pred = np.concatenate(pred_list, 0)

            t = np.abs(hist_score - pred) < thresh
            F = np.sum(t) / len(hist_score)
            fidelity_list.append(F)
        model.train()
        print(fidelity_list)
        with open("../vis/%s/%s_fidelity.txt" % (args.data_set, args.data_set), 'a') as f:
            f.write(str(fidelity_list) + '\n')


def fidelity_compare(dataset):
    with open("../vis/%s/%s_draw.txt" % (dataset, dataset), 'r') as f:
        lines = f.readlines()
    y0 = eval(lines[0])
    y1 = eval(lines[1])
    y2 = eval(lines[2])
    y3 = eval(lines[3])
    y4 = eval(lines[4])
    x = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    plt.plot(x, y0, 'cyan', label='HAKT(0)')
    plt.plot(x, y1, 'green', label='HAKT(0.1)')
    plt.plot(x, y2, 'orange', label='HAKT(0.2)')
    plt.plot(x, y3, 'blue', label='HAKT(0.3)')
    plt.plot(x, y4, 'red', label='SAKT')
    plt.legend()
    plt.grid()
    plt.xlabel("Threshold value for explainability")
    plt.ylabel("Fidelity")
    # plt.show()
    plt.title('%s' % dataset)
    plt.savefig("../vis/%s/%s_fidelity_compare.png" % (dataset, dataset))
    plt.clf()