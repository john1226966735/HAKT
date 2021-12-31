from abc import ABC
import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import graphviz

from code.model import KT_Model


class VisTools(nn.Module, ABC):
    def __init__(self, loader, args):
        super(VisTools, self).__init__()
        self.loader = loader
        self.data_set = args.data_set
        self.meta_paths = args.meta_paths

        # load pre-trained model
        device = torch.device(args.device)
        self.model = KT_Model(args, device).to(device)

        if args.data_set == 'ASSIST09':
            param_file = "params_ASSIST09_128_2021-11-12&06-04-55_0.7880.pkl"
        elif args.data_set == 'ASSIST12':
            param_file = "params_ASSIST12_128_2021-09-13&18-33-31_0.7677.pkl"
        elif args.data_set == 'ASSIST17':
            param_file = "params_ASSIST17_128_2021-09-29&22-03-38_0.7603.pkl"
        elif args.data_set == 'EdNet':
            param_file = "params_EdNet_128_2021-09-14&20-26-01_0.7596.pkl"
        elif args.data_set == 'Eedi':
            param_file = "params_Eedi_128_2021-09-14&10-52-15_0.7882.pkl"
        elif args.data_set == 'Statics2011':
            param_file = "params_Statics2011_128_2021-09-29&18-43-19_0.8832.pkl"
        else:
            param_file = None
        state_dict = torch.load("../param/%s/%s" % (self.data_set, param_file))
        for k in state_dict.keys():
            print(k)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("load model done!")
        self.flag = None
        self.file_id = None

    @staticmethod
    def group_topk(df, groups, group_by='dst_node', sort_by='elem_weight', k=3):
        df_list = []
        for g in groups:
            df_list.append(df[df[group_by] == g])
        df1_list = []
        for df in df_list:
            df1_list.append(df.nlargest(k, sort_by, keep='all'))
        df2 = pd.concat(df1_list, axis=0)
        return df2

    @staticmethod
    def process_print(record_df, hist_semantic_df, hist_element_df_list, next_element_df_list, next_semantic_df,
                      next_ques_id, predict, label):
        # print 
        for df in [record_df, hist_semantic_df]:
            line = []
            for ind, row in df.iterrows():
                line.append(tuple([row[name] for name in df.columns]))
            print(line)

        for element_df in hist_element_df_list:
            line = []
            for ind, row in element_df.iterrows():
                line.append((row['dst_node'], row['scr_node'], row['elem_weight']))
            print(line)

        for element_df in next_element_df_list:
            line = []
            for ind, row in element_df.iterrows():
                line.append((row['dst_node'], row['scr_node'], row['elem_weight']))
            print(line)

        line = []
        for ind, row in next_semantic_df.iterrows():
            line.append(tuple([row[name] for name in df.columns]))
        print(line)

        print(next_ques_id, predict, label)

    def process_viz(self, record_df, hist_semantic_df, hist_element_df_list,
                    next_element_df_list, next_semantic_df,
                    next_ques_id, predict, label):

        f = graphviz.Digraph('dot', 'process_visualize', filename='../vis/%s_%s_pv.gv' % (self.data_set, self.flag))
        f.attr('node', shape='circle')

        for ind, row in record_df.iterrows():
            f.node('q%d' % row['question_id'], label='q%d(%d)' % (row['question_id'], row['answer']))
            f.edge('u1', 'q%d' % row['question_id'], label='%.3f' % row['record_weight'])

        for ind, row in hist_semantic_df.iterrows():
            for i, t in enumerate(self.meta_paths):
                f.edge('q%d' % row['question_id'], '%s(%s)' % (t, 'q%d' % row['question_id']),
                       label='%.3f' % row['semantic%d' % i])

        for t, element_df in zip(self.meta_paths, hist_element_df_list):
            for ind, row in element_df.iterrows():
                f.edge('%s(%s)' % (t, 'q%d' % row['dst_node']), '%s%d' % (t[1:], row['scr_node']),
                       label='%.3f' % row['elem_weight'])

        for t, element_df in zip(self.meta_paths, next_element_df_list):  # 每个element_df对应一种语义
            for ind, row in element_df.iterrows():
                f.edge('%s%d' % (t[1:], row['scr_node']), '%s(%s)' % (t, 'q%d' % row['dst_node']),
                       label='%.3f' % row['elem_weight'])

        for ind, row in next_semantic_df.iterrows():
            f.node('q%d' % row['question_id'], label='q%d(%.3f_%d)' % (row['question_id'], predict, label))
            for i, t in enumerate(self.meta_paths):
                f.edge('%s(%s)' % (t, 'q%d' % row['question_id']), 'q%d' % row['question_id'],
                       label='%.3f' % row['semantic%d' % i])

        f.view()

    def case_study_step(self, params_dict):
        # path_weight[t, num_path], attn_weight: [t], answer:[t], predict: [t]
        for k, v in params_dict.items():
            if isinstance(v, list):
                params_dict[k] = [i.cpu().detach().numpy() for i in v]
            else:
                params_dict[k] = v.cpu().detach().numpy()

        record_weight = params_dict['record_weight']
        semantic_weight = params_dict['semantic_weight']
        element_weight_list = [np.concatenate([edge.transpose(1, 0), weight], axis=1) for edge, weight in
                               zip(params_dict['edges'], params_dict['element_weights'])]

        ques_id = params_dict['ques_id']
        answer = params_dict['answer']
        next_ques_id = params_dict['index'][-1]
        predict = params_dict['predict'][-1]
        label = params_dict['label'][-1]

        record_df = pd.DataFrame({'question_id': ques_id, 'record_weight': record_weight, 'answer': answer})
        record_df = record_df.nlargest(5, 'record_weight', keep='all')
        relate_questions = list(record_df['question_id'])

        semantic_cols = ['semantic' + str(k) for k in range(semantic_weight.shape[1])]
        hist_semantic_df = pd.DataFrame(semantic_weight[relate_questions, :], columns=semantic_cols)
        hist_semantic_df['question_id'] = relate_questions
        next_semantic_df = pd.DataFrame(np.expand_dims(semantic_weight[next_ques_id, :], axis=0), columns=semantic_cols)
        next_semantic_df['question_id'] = next_ques_id

        element_cols = ['scr_node', 'dst_node', 'elem_weight']
        element_df_list = [pd.DataFrame(elem_weight, columns=element_cols) for elem_weight in element_weight_list]
        hist_element_df_list = [
            self.group_topk(element_df[element_df['dst_node'].isin(relate_questions)], relate_questions) for element_df
            in element_df_list]
        next_element_df_list = [
            element_df[element_df['dst_node'] == next_ques_id].nlargest(5, 'elem_weight', keep='all') for element_df in
            element_df_list]

        # self.process_print(record_df, hist_semantic_df, hist_element_df_list, next_element_df_list, next_semantic_df,
        #                  next_ques_id, predict, label)

        self.process_viz(record_df, hist_semantic_df, hist_element_df_list, next_element_df_list, next_semantic_df,
                         next_ques_id, predict, label)

    def case_study(self):  # 1. 案例分析(决策过程分析)：说明模型根据权重大的历史题目的答题情况做预测（一个序列样本的一个时间步）
        print("case study...")
        for i, (seq_lens, pad_data, pad_answer, pad_index, pad_label) in enumerate(self.loader['test']):
            # if i < 3:
            #     continue
            pad_predict = self.model(pad_data, pad_answer, pad_index)  # [L, B]

            for b in [10]:  # 取一个序列样本
                # print("##########################################")
                seq_len = seq_lens.cpu().detach().numpy()[b]
                for t in [min(20, seq_len)]:  # 取一个时间步
                    params_dict = {'ques_id': pad_data[:t, b], 'answer': pad_answer[:t, b],
                                   'index': pad_index[:t, b],
                                   'label': pad_label[:t, b], 'predict': pad_predict[:t, b],
                                   'edges': self.model.QuesEmb_Layer.edge_list,
                                   'element_weights': [torch.mean(weight.squeeze(-1), dim=1, keepdim=True)
                                                       for weight in self.model.QuesEmb_Layer.element_weights],
                                   'semantic_weight': self.model.QuesEmb_Layer.semantic_weight.squeeze(),
                                   'record_weight': torch.mean(self.model.attn_weight, 1, False)[b, t - 1, :t]}
                    self.flag = "%d_%d_%d_g4" % (i, b, t)
                    self.case_study_step(params_dict)
            break
