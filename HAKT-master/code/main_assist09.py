import argparse
import numpy as np
from code.train import train
from code.loader import load_data
from code.Exp_calculate_fidelity import get_fidelity
from code.Exp_decision_process import visualize


def parse_args():
    parser = argparse.ArgumentParser()
    '''
    question embedding module
    '''
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--num_ques", type=int, default=15680)
    parser.add_argument("--gat_heads", type=int, default=[[4, 4, 4], [4, 4, 4]])
    parser.add_argument("--meta_paths", type=list, default=['qk', 'qt'])
    parser.add_argument("--fusion", type=str, default='attnVec_nonLinear')

    '''
    knowledge state module
    '''
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_attn_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--drop_prob", type=int, default=0.05)
    parser.add_argument("--attention_mode", type=str, default='general')

    '''
    prediction module
    '''
    parser.add_argument("--predict_mode", type=str, default="2")  # '1', '2'
    parser.add_argument("--predict_type", type=str, default="mlp")  # dot, mlp
    parser.add_argument("--num_hidden_layer", type=int, default=1)

    '''
    loader
    '''
    parser.add_argument("--min_seq_len", type=int, default=3)
    parser.add_argument("--max_seq_len", type=int, default=200)
    parser.add_argument("--input", type=str, default='question')
    parser.add_argument("--data_path", type=str, default="../data") 
    parser.add_argument("--data_set", type=str, default='ASSIST09')

    '''
    train
    '''
    parser.add_argument("--l2_weight", type=float, default=1e-4)
    parser.add_argument("--explain_wgt", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_decay", type=float, default=1)
    parser.add_argument("--max_epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument('--device', type=str, default="cuda:1")

    '''
    others
    '''
    parser.add_argument('--save_dir', type=str,
                        default='../result/ASSIST09', help='the dir which save results')
    parser.add_argument('--log_file', type=str,
                        default='logs.txt', help='the name of logs file')
    parser.add_argument('--result_file', type=str,
                        default='tunings.txt', help='the name of results file')
    parser.add_argument('--remark', type=str,
                        default='', help='remark the experiment')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    data_loader = load_data(args)

    # choose one of the following operations
    train(data_loader, args)  # train the model and get AUC
    # visualize(data_loader, args)  # visualize the model's decision process
    # get_fidelity(data_loader, args)  # calculate the model's fidelity
