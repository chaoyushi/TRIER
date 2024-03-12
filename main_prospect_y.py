import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
from trier_pt import TRIER_PT
from trier_rt import TRIER_RT
from dataset_duorec import TrainPTDataset, TestDataset
from script import *
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./tensorboard_log')


def get_metric(epoch, total_result):
    total_result_dict = {'epoch': epoch}
    new_result = []
    # 全集
    total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
    total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
    total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
    total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
    total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
    total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
    total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
    total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
    total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
    # ILD
    total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
    total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
    total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)

    return total_result_dict

def metric_all_intervals(epoch, total_result, length = None):
    # total_result_dict = {'epoch': epoch}

    if length != None:
        length_lower_bound = [0, 10, 20, 30, 40]
        length_upper_bound = [10, 20, 30, 40, 51]

        for ldx in range(len(length_lower_bound)):
            filter_pred_list = []
            for i in range(len(total_result)):
                if length_lower_bound[ldx] <= length[i] and length[i] < length_upper_bound[ldx]:
                    filter_pred_list.append(total_result[i])
            print("input length 长度:", length_lower_bound[ldx],"-", length_upper_bound[ldx], get_metric(epoch, filter_pred_list))

    return get_metric(epoch, total_result)

def metric_all(epoch, total_result, total_rec_set=None, cate_map=None, num_cat=None):
    total_result_dict = {'epoch': epoch}

    # 全集
    total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
    total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
    total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
    total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
    total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
    total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
    total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
    total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
    total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
    # ILD
    total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
    total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
    total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)
    #
    # total_result_dict['CC@5'] = get_metrics_full('CC@5', total_result)
    # total_result_dict['CC@10'] = get_metrics_full('CC@10', total_result)
    # total_result_dict['CC@20'] = get_metrics_full('CC@20', total_result)
    # Coverage
    if total_rec_set is not None:
        total_result_dict['CC@5'] = get_coverage('CC@5', total_rec_set, cate_map, num_cat)
        total_result_dict['CC@10'] = get_coverage('CC@10', total_rec_set, cate_map, num_cat)
        total_result_dict['CC@20'] = get_coverage('CC@20', total_rec_set, cate_map, num_cat)

    return total_result_dict


if __name__ == '__main__':


    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_file = args.tf
    valid_file = args.vf
    test_file = args.ef
    valid_neg_file = args.vn
    test_neg_file = args.en
    batch_size = args.b
    log_step = args.ls
    learning_rate = args.l
    epochs = args.e
    dropout_rate = args.dr
    hidden_unit = args.hd
    head_num = args.hn
    layer_num = args.ln
    load_path = args.i
    save_path = args.o
    mode = args.m
    resume = args.r
    item_num = args.n
    max_seqs_len = args.ml
    modified_max_seqs_len = args.mml
    cate_file = args.cat
    num_cat = args.n_cat

    item2vec = np.load("./Yelp/yelp_vec.npy")# Yelp

    item2vec = torch.tensor(item2vec)
    cate_map = get_cates_map(cate_file)
   
    rt_model = TRIER_RT(item_num, 2, head_num, hidden_unit, dropout_rate, batch_size, args)
    
    rt_model.load_state_dict(torch.load(load_path + 'model/duorec-' + str(103) + '.pth', map_location=args.device))
    for param in rt_model.parameters():
        param.requires_grad = False
    # init_seeds(seed=4444)
    init_seeds()
    if mode == 'train':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'model/'):
            os.makedirs(save_path + 'model/')
        if resume:
            fw = open(save_path + 'train_result.txt', 'a')
        else:
            fw = open(save_path + 'train_result.txt', 'w')
        dataset = TrainPTDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

        last_epoch = 0
        if resume:
            with open(save_path + 'train_result.txt', 'r') as f:
                content = f.readlines()
            last_epoch =  111# len(content)
            print('load model：epoch %d' % (last_epoch,))
            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(last_epoch) + '.pth', map_location=args.device))
        else:
            print('initialize model')
            model.apply(xavier_init)
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epoch = last_epoch
        while epoch < epochs:
            epoch += 1
            step = 0
            loss_avg = 0
            loss_acc, loss_div, loss_nce = 0.0, 0.0, 0.0
            start_time = time.time()
            for batch in dataloader:
                step += 1
                optimizer.zero_grad()
                input_session_ids, targets, negatives, sem_aug_input_session_ids, input_reverse_ids = batch
                if torch.cuda.is_available():
                    input_session_ids = input_session_ids.cuda()
                    targets = targets.cuda()
                    negatives = negatives.cuda()
                    sem_aug_input_session_ids = sem_aug_input_session_ids.cuda()
                    input_reverse_ids = input_reverse_ids.cuda()
                    # sem_aug_reverse_session_ids = sem_aug_reverse_session_ids.cuda()
                output, nce_loss, div_loss = model.train_forward(input_session_ids, sem_aug_input_session_ids,
                                            input_reverse_ids, rt_model, item2vec)
                loss, main_loss = model.rec_loss(output, targets, nce_loss, div_loss)
                loss.backward()
                optimizer.step()
                loss_avg += loss
                loss_acc += main_loss
                loss_div += div_loss
                loss_nce += nce_loss
                if step % log_step == 0:
                    print('epoch %d step %d loss %0.4f time %d' % (epoch, step, loss_avg / step, time.time()-start_time))

            torch.save(model.state_dict(), save_path + 'model/duorec-' + str(epoch) + '.pth')
            print('epoch %d loss %0.4f time %d' % (epoch, loss_avg / step, time.time() - start_time))
            fw.write('epoch %d loss %0.4f' % (epoch, loss_avg / step) + '\n')
        fw.close()

    result_list = []
    if mode == "valid":
        if resume:
            fw = open(save_path + str(mode) + '_result.txt', 'a')
        else:
            fw = open(save_path + str(mode) + '_result.txt', 'w')
        dataset = TestDataset(valid_file, valid_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()
        model.eval()
        next_epoch = 1# 50 # 83
        if resume:
            with open(save_path + str(mode) + '_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print(str(mode) + ' from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []

            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(epoch) + '.pth', map_location=args.device)) # , map_location=torch.device('cpu')
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, input_reverse_ids = batch
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        input_reverse_ids = input_reverse_ids.cuda()
                    if args.t_mode == "topk":
                        # 一次性生成TopK
                        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
                        output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    elif args.t_mode == "greedy":
                        # step by step 生成全集
                        output, rec_list = model.test_forward(input_session_ids, input_reverse_ids, rt_model, True)
                        # output, rec_list = model.test_forward_decoder(input_session_ids)
                    else:
                        # 通过encoder生成的模块进行评测，不使用decoder
                        output = model.test_forward(input_session_ids)  # [batch_size, hidden_unit]
                        output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    # 评价所有指标，包括采样集，全集，ILD
                    result = evaluate_function_with_full(targets, rec_list, item2vec=item2vec)
                    total_result.extend(result)

            total_result_dict = metric_all(epoch, total_result)
            result_list.append(total_result_dict)
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            # time.sleep(80)
            with open(save_path + str(mode) + '_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()
        # find best valid epoch and test result
        def get_best_epoch(score):
            epcoh = 0
            max_r5 = 0.0
            for item in score:
                recall = item["recall@5_f"]  # .float()
                if recall >= max_r5:
                    max_r5 = item["recall@5_f"]
                    epcoh = item["epoch"]
            return max_r5, epcoh


        print("验证集中recall最高指，以及对应的epoch", get_best_epoch(result_list))

    if mode == "test":
        if resume:
            fw = open(save_path + 'test_result.txt', 'a')
        else:
            fw = open(save_path + 'test_result.txt', 'w')
        dataset = TestDataset(test_file, test_neg_file, item_num, modified_max_seqs_len) if mode == "test" \
            else TestDataset(valid_file, valid_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = TRIER_PT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
        if torch.cuda.is_available():
            model.cuda()
            rt_model.cuda()
        model.eval()
        next_epoch = 128 # 50 # 83 273  182
        if resume:
            with open(save_path + str(mode) + '_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print(str(mode) + ' from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []  # 保存用于计算Recall， MRR， NDCG，ILD的指标
            total_length = []
            total_rec_set = [set(), set(), set()]  # 保存rec list用于计算Coverage指标

            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(epoch) + '.pth', map_location=args.device)) # , map_location=torch.device('cpu')
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, input_reverse_ids = batch
                    item_seq_len = (input_session_ids > 0).sum(-1).tolist()
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                        input_reverse_ids = input_reverse_ids.cuda()
                    if args.t_mode == "topk": # 一次性生成TopK
                        output = model.test_forward(input_session_ids, input_reverse_ids, rt_model, False)
                        output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                        _, rec_list = output.log_softmax(-1).topk(k=20, axis=-1)
                    elif args.t_mode == "greedy": # step by step 生成全集
                        output, rec_list = model.test_forward(input_session_ids, input_reverse_ids, rt_model, True)
                        # output, rec_list = model.test_forward_decoder(input_session_ids)
                    else:
                        pass
                    # 评价所有指标，包括采样集，全集，ILD
                    result = evaluate_function_with_full(targets, rec_list, item2vec=item2vec)#, cat_map=cate_map, cat_num=num_cat)
                    total_rec_set = get_coverage_set(total_rec_set, rec_list)
                    total_result.extend(result)

                    total_length.extend(item_seq_len)

            total_result_dict = metric_all_intervals(epoch, total_result, total_length)
            # total_result_dict = metric_all(epoch, total_result)
            # total_result_dict = metric_all(epoch, total_result, total_rec_set=total_rec_set,
            #                                cate_map=cate_map, num_cat=num_cat)

            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            # time.sleep(90)
            with open(save_path + str(mode) + '_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()

