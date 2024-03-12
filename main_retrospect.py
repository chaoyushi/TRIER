import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
from trier_rt import TRIER_RT
from dataset_duorec import TrainRTDataset, TestDataset
from script import *

# def metric_all(epoch, total_result, total_rec_set=None, cate_map=None, num_cat=None):
#     total_result_dict = {'epoch': epoch}
#
#     # 全集
#     total_result_dict['recall@5_f'] = get_metrics_full('recall@5_f', total_result)
#     total_result_dict['recall@10_f'] = get_metrics_full('recall@10_f', total_result)
#     total_result_dict['recall@20_f'] = get_metrics_full('recall@20_f', total_result)
#     total_result_dict['mrr@5_f'] = get_metrics_full('mrr@5_f', total_result)
#     total_result_dict['mrr@10_f'] = get_metrics_full('mrr@10_f', total_result)
#     total_result_dict['mrr@20_f'] = get_metrics_full('mrr@20_f', total_result)
#     total_result_dict['ndcg@5_f'] = get_metrics_full('ndcg@5_f', total_result)
#     total_result_dict['ndcg@10_f'] = get_metrics_full('ndcg@10_f', total_result)
#     total_result_dict['ndcg@20_f'] = get_metrics_full('ndcg@20_f', total_result)
#     # ILD
#     total_result_dict['ILD@5'] = get_metrics_full('ILD@5', total_result)
#     total_result_dict['ILD@10'] = get_metrics_full('ILD@10', total_result)
#     total_result_dict['ILD@20'] = get_metrics_full('ILD@20', total_result)

    # total_result_dict['CC@5'] = get_metrics_full('CC@5', total_result)
    # total_result_dict['CC@10'] = get_metrics_full('CC@10', total_result)
    # total_result_dict['CC@20'] = get_metrics_full('CC@20', total_result)
    # Coverage
    # if total_rec_set is not None:
    #     total_result_dict['CC@5'] = get_coverage_score('CC@5', total_rec_set, cate_map, num_cat)
    #     total_result_dict['CC@10'] = get_coverage_score('CC@10', total_rec_set, cate_map, num_cat)
    #     total_result_dict['CC@20'] = get_coverage_score('CC@20', total_rec_set, cate_map, num_cat)
        # total_result_dict['CC@5'] = get_coverage('CC@5', total_rec_set, cate_map, num_cat)
        # total_result_dict['CC@10'] = get_coverage('CC@10', total_rec_set, cate_map, num_cat)
        # total_result_dict['CC@20'] = get_coverage('CC@20', total_rec_set, cate_map, num_cat)

    # return total_result_dict
def metric_all(epoch, total_result):
    total_result_dict = {'epoch': epoch}
    # 采样集
    total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
    total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
    total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
    total_result_dict['mrr@5'] = get_metrics('mrr@5', total_result)
    total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
    total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
    total_result_dict['ndcg@5'] = get_metrics('ndcg@5', total_result)
    total_result_dict['ndcg@10'] = get_metrics('ndcg@10', total_result)
    total_result_dict['ndcg@20'] = get_metrics('ndcg@20', total_result)

    total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                               total_result_dict['recall@20'] + total_result_dict['mrr@5'] + \
                               total_result_dict['mrr@10'] + total_result_dict['mrr@20']

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
    save_path = args.o
    mode = args.m
    resume = args.r
    item_num = args.n
    max_seqs_len = args.ml
    modified_max_seqs_len = args.mml

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
        dataset = TrainRTDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model = TRIER_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)

        last_epoch = 0
        if resume:
            with open(save_path + 'train_result.txt', 'r') as f:
                content = f.readlines()
            last_epoch = 79 # len(content)
            print('load model：epoch %d' % (last_epoch,))
            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(last_epoch) + '.pth', map_location=args.device))
        else:
            print('initialize model')
            model.apply(xavier_init)
        if torch.cuda.is_available():
            model.cuda()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epoch = last_epoch
        while epoch < epochs:
            epoch += 1
            step = 0
            acc_loss = 0
            start_time = time.time()
            for batch in dataloader:
                step += 1
                optimizer.zero_grad()
                input_session_ids, targets, negatives, sem_aug_input_session_ids = batch
                if torch.cuda.is_available():
                    input_session_ids = input_session_ids.cuda()
                    targets = targets.cuda()
                    negatives = negatives.cuda()
                    sem_aug_input_session_ids = sem_aug_input_session_ids.cuda()
                output, nce_loss, dis_reg, me_reg= model.train_forward(input_session_ids, sem_aug_input_session_ids)
                loss = model.rec_loss(output, targets, nce_loss, dis_reg, me_reg)
                loss.backward()
                optimizer.step()
                acc_loss += loss
                if step % log_step == 0:
                    print('epoch %d step %d loss %0.4f time %d' % (
                    epoch, step, acc_loss / step, time.time()-start_time))
            torch.save(model.state_dict(), save_path + 'model/duorec-' + str(epoch) + '.pth')
            print('epoch %d loss %0.4f time %d' % (
            epoch, acc_loss / step, time.time() - start_time))
            fw.write('epoch %d loss %0.4f' % (epoch, acc_loss / step) + '\n')
        fw.close()

    if mode == "valid":
        if resume:
            fw = open(save_path + 'valid_result.txt', 'a')
        else:
            fw = open(save_path + 'valid_result.txt', 'w')
        dataset = TestDataset(valid_file, valid_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = TRIER_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args=args)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = 1
        if resume:
            with open(save_path + 'valid_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('valid from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []
            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(epoch) + '.pth', map_location=args.device))
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, _ = batch
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                    output = model.test_forward(input_session_ids)  # [batch_size, hidden_unit]
                    output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                    _, output_token = output.log_softmax(-1).topk(k=20, axis=-1)
                    # result = evaluate_function(output, targets, negatives)
                    # print(output_token.shape)
                    result = evaluate_function_with_full(targets, output_token)
                    # result = evaluate_function_with_full(output, targets, negatives, output_token)

                    total_result.extend(result)
            # print("step 1 done!")
            total_result_dict = metric_all(epoch, total_result)
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')

            with open(save_path + 'valid_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()

    if mode == "test":
        if resume:
            fw = open(save_path + 'test_result.txt', 'a')
        else:
            fw = open(save_path + 'test_result.txt', 'w')
        dataset = TestDataset(test_file, test_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = TRIER_RT(item_num, layer_num, head_num, hidden_unit, dropout_rate, batch_size, args)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = 1
        if resume:
            with open(save_path + 'test_result.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('test from epoch %d' % (next_epoch,))
        epoch = next_epoch
        import time
        while epoch <= epochs:
            # time.sleep(17)
            step = 0
            total_result = []
            model.load_state_dict(torch.load(save_path + 'model/duorec-' + str(epoch) + '.pth', map_location=args.device)) # , map_location=torch.device('cpu')
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    input_session_ids, targets, negatives, _ = batch
                    if torch.cuda.is_available():
                        input_session_ids = input_session_ids.cuda()
                        targets = targets.cuda()
                        negatives = negatives.cuda()
                    output = model.test_forward(input_session_ids)  # [batch_size, hidden_unit]
                    output = torch.matmul(output, model.item_embedding.weight.T)  # [batch_size, item_num]
                    result = evaluate_function(output, targets, negatives)
                    # _, output_token = output.log_softmax(-1).topk(k=20, axis=-1)
                    # result = evaluate_function_with_full(output, targets, negatives, output_token)
                    # print(result)
                    # break
                    total_result.extend(result)
                    # break

            total_result_dict = metric_all(epoch, total_result)
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            with open(save_path + 'test_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')
            epoch += 1
        fw.close()
