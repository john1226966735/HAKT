import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

import numpy as np
from sklearn import metrics

from code.train_utils import Logger
from code.model import KT_Model


def train(loader, args):
    logger = Logger(args)
    device = torch.device(args.device)
    model = KT_Model(args, device).to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_weight)

    criterion = nn.BCELoss(reduction='mean')

    for epoch in range(1, args.max_epoch + 1):
        logger.epoch_increase()
        for i, (seq_lens, pad_data, pad_answer, pad_index, pad_label) in enumerate(loader['train']):
            pad_predict = model(pad_data, pad_answer, pad_index)  # 运行模型
            pack_predict = pack_padded_sequence(pad_predict, seq_lens, enforce_sorted=True)
            pack_label = pack_padded_sequence(pad_label, seq_lens, enforce_sorted=True)

            explain_loss = model.explain_loss(pad_answer, pad_predict)

            pred_loss = criterion(pack_predict.data, pack_label.data)
            loss = (1 - args.explain_wgt) * pred_loss + args.explain_wgt * explain_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_metrics_dict = evaluate(model, loader['train'])
        test_metrics_dict = evaluate(model, loader['test'])

        logger.one_epoch(epoch, train_metrics_dict, test_metrics_dict, model)

        if logger.is_stop():
            break
        # end of epoch
    logger.one_run(args)
    # end of run


def evaluate(model, data):
    model.eval()
    true_list, pred_list = [], []
    for seq_lens, pad_data, pad_answer, pad_index, pad_label in data:
        pad_predict = model(pad_data, pad_answer, pad_index)  # 运行模型
        pack_predict = pack_padded_sequence(pad_predict, seq_lens, enforce_sorted=True)
        pack_label = pack_padded_sequence(pad_label, seq_lens, enforce_sorted=True)

        y_true = pack_label.data.cpu().contiguous().view(-1).detach()
        y_pred = pack_predict.data.cpu().contiguous().view(-1).detach()

        true_list.append(y_true)
        pred_list.append(y_pred)
    auc = metrics.roc_auc_score(np.concatenate(true_list, 0), np.concatenate(pred_list, 0))
    model.train()
    return {'auc': auc}
