import random
import numpy as np
import torch
import argparse
from torch import nn
from torch.backends import cudnn
import torch.nn.functional as F
from tqdm import tqdm

# item2vec = torch.load("item_emb.pt")# Beauty
# item2vec = np.load("./Beauty-m/beauty_emb_12102.npy") # Beauty
# item2vec = np.load("./Yelp/yelp_vec.npy")# Yelp
# item2vec = np.load("./Steam/item_emb_steam.npy")
# item2vec = np.load("./Steam17/item_emb_steam.npy")
# item2vec = np.load("./Sports_and_Outdoors/item_emb_sport.npy")  # Sports
# item2vec = np.load("./ML1M/ml-1m_vec.npy") # ML1M
# cate_file = "./Beauty-m/beauty_cate.txt"
# cate_file = "./Yelp/yelp_cate.txt"
# item2vec = torch.tensor(item2vec)

# 获取map
# iidcate_map = get_cates_map(cate_file)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tf', type=str, default='./Beauty-m/train_beauty.dat', help='训练集数据的路径')
    parser.add_argument('-vf', type=str, default='./Beauty-m/valid_beauty.dat', help='验证集数据的路径')
    parser.add_argument('-ef', type=str, default='./Beauty-m/test_beauty.dat', help='测试集数据的路径')
    parser.add_argument('-cat', type=str, default="./Beauty-m/beauty_cate.txt", help='catergory数据路径')
    parser.add_argument('-n_cat', type=int, default=656, help='catergory数量')
    parser.add_argument('-vn', type=str, default='./Beauty-m/Beauty-random-sample_size=99-seed=4444.txt', help='验证集负例的路径')
    parser.add_argument('-en', type=str, default='./Beauty-m/Beauty-random-sample_size=99-seed=4444.txt', help='测试集负例的路径')

    parser.add_argument('-b', type=int, default=256, help='批量大小')
    parser.add_argument('-bt', type=int, default=256, help='train批量大小')
    parser.add_argument('-bv', type=int, default=256, help='valid and test 批量大小')
    parser.add_argument('-ls', type=int, default=50, help='log_step')
    parser.add_argument('-l', type=float, default=1e-3, help='学习率')
    parser.add_argument('-e', type=int, default=300, help='训练轮数')
    parser.add_argument('-dr', type=float, default=0.5, help='正则化')
    parser.add_argument('-hd', type=int, default=64, help='隐藏层维度')
    parser.add_argument('-hn', type=int, default=2, help='多头注意力机制中头的数量')
    parser.add_argument('-ln', type=int, default=2, help='transformer的层数')
    parser.add_argument('-o', type=str, default='./save_duorec/', help='保存路径')
    parser.add_argument('-i', type=str, default='./save_duorec/', help='加载路径')
    parser.add_argument('-m', type=str, default="train", help='训练，验证或测试')
    parser.add_argument('-r', action='store_true', help='是否继续')
    parser.add_argument('-n', type=int, default=12102, help='商品数量')  # Beauty数据集有12101个item +3是因为padding EOS mask
    parser.add_argument('-ml', type=int, default=50, help='max_seqs_len')
    parser.add_argument('-mml', type=int, default=72, help='modified_max_seqs_len')
    parser.add_argument('-bw', type=int, default=3, help='beam width')
    parser.add_argument('-k', type=int, default=5, help='length of recommendation list')
    parser.add_argument('-div', action='store_true', help='是否使用div loss')
    parser.add_argument('-kl', action='store_true', help='是否使用kl loss')
    parser.add_argument('-warmup', action='store_true', help='是否使用warm up')
    parser.add_argument('-reg', action='store_true', help='是否在retrospect阶段使用正则化 loss')
    parser.add_argument('-lamb',  type=float, default=0.5, help='准确性评分和多样性评分之间的权衡参数')
    parser.add_argument('-t_mode', type=str, default="greedy", help='topk:使用第一个step的logit去预测topk, '
                                                    'greedy: step by step生成')
    parser.add_argument('-ssl', type=str, default="us_x", help='默认使用us_x, 设置为none不适用CL loss')

    parser.add_argument('-lm', type=int, default=5, help='向左增广数量，默认设置为5')

    args = parser.parse_args()

    return args

def get_cates_map(cate_file):
    iidcate_map = {}  #iid:cates
    ## movie_id:cate_ids, cate_ids is not only one
    with open(cate_file) as f_cate:
        for l in f_cate.readlines():
            if len(l) == 0: break
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')]
            iid, cate_ids = items[0], items[1:]
            iidcate_map[iid] = cate_ids
    return iidcate_map


def get_coverage_set(total_rec_set, rec_list):
    # coverage = (rec_list[:,:5], rec_list[:,:10], rec_list)
    # print(rec_list.shape, rec_list[:,:5].shape, rec_list[:,:10].shape)
    total_rec_set[0] = total_rec_set[0].union(set(rec_list[:,:5].contiguous().view(-1,1).squeeze(1).tolist()))
    total_rec_set[1] = total_rec_set[1].union(set(rec_list[:,:10].contiguous().view(-1,1).squeeze(1).tolist()))
    total_rec_set[2] = total_rec_set[2].union(set(rec_list.contiguous().view(-1,1).squeeze(1).tolist()))

    return total_rec_set



def coverage_for_user(u_set, cat_map, cat_num):
    cc = set()
    for item in u_set:
        for cat in cat_map[item]:
            cc.add(cat)
    return len(cc)/cat_num


def ILD_tensor(token_list, item2vec):
    # ILD = 0
    topk = token_list.shape[1]
    gen_vector = item2vec[token_list]

    # B X N x N
    ILD_list = torch.cdist(gen_vector, gen_vector)/(topk*(topk - 1))
    # print(token_list)
    # print(gen_vector)
    # print(ILD_list)
    # print(ILD_list.sum(-1).sum(-1))
    return ILD_list.sum(-1).sum(-1)  # B

def init_seeds(seed=0, cuda_deterministic=True):
    """
    初始化随机数种子
    :param seed: 随机数种子
    :param cuda_deterministic: 是否固定cuda的随机数种子
    设置这个 flag 为True，我们就可以在 PyTorch 中对模型里的卷积层进行预先的优化
    如果我们的网络模型一直变的话，不能设置cudnn.benchmark=True。因为寻找最优卷积算法需要花费时间。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def xavier_init(model):
    """
    初始化模型参数
    :param model: 给定模型
    """
    for name, par in model.named_parameters():
        if 'weight' in name and len(par.shape) >= 2:
            nn.init.xavier_normal_(par)
        elif 'bias' in name:
            nn.init.constant_(par, 0.0)


def evaluate_function_with_full(positives, output_token,
                                cat_map=None, cat_num=None, item2vec=None):  # 增加一个recall@20 去掉auc mrr改成mrr@5 mrr@10 mrr@20
    result = [{} for _ in range(len(output_token))]
    # 全集评测score
    # output_token = output_token.tolist()
    for i in range(len(output_token)):
        # 全集指标
        gen_list = output_token[i]
        answer = positives[i]
        idx = torch.nonzero(gen_list == answer)

        # print(idx)
        if len(idx) == 0:
            result[i]['recall@5_f'] = 0.0
            result[i]['mrr@5_f'] = 0.0
            result[i]['ndcg@5_f'] = 0.0
            result[i]['recall@10_f'] = 0.0
            result[i]['mrr@10_f'] = 0.0
            result[i]['ndcg@10_f'] = 0.0
            result[i]['recall@20_f'] = 0.0
            result[i]['mrr@20_f'] = 0.0
            result[i]['ndcg@20_f'] = 0.0
        else:
            idx = idx.item()
            if idx < 5 :
                result[i]['recall@5_f'] = 1.0
                result[i]['mrr@5_f'] = 1 / (idx + 1)
                result[i]['ndcg@5_f'] = 1.0 / np.log2((idx + 1) + 2.0)
            else:
                result[i]['recall@5_f'] = 0.0
                result[i]['mrr@5_f'] = 0.0
                result[i]['ndcg@5_f'] = 0.0
            if idx < 10 :
                result[i]['recall@10_f'] = 1.0
                result[i]['mrr@10_f'] = 1 / (idx + 1)
                result[i]['ndcg@10_f'] = 1.0 / np.log2((idx + 1) + 2.0)
            else:
                result[i]['recall@10_f'] = 0.0
                result[i]['mrr@10_f'] = 0.0
                result[i]['ndcg@10_f'] = 0.0
            if idx < 20:
                result[i]['recall@20_f'] = 1.0
                result[i]['mrr@20_f'] = 1 / (idx + 1)
                result[i]['ndcg@20_f'] = 1.0 / np.log2((idx + 1) + 2.0)
            else:
                result[i]['recall@20_f'] = 0.0
                result[i]['mrr@20_f'] = 0.0
                result[i]['ndcg@20_f'] = 0.0
        gen_list = output_token[i].cpu().numpy() #% 12103
        # gen_list = torch.add(output_token[i], 1).cpu().numpy()%12103
        # print(item2vec.shape, gen_list.shape)
        result[i]['ILD@5'] = cal_ILD(item2vec[gen_list[:5]], 5)
        result[i]['ILD@10'] = cal_ILD(item2vec[gen_list[:10]], 10)
        result[i]['ILD@20'] = cal_ILD(item2vec[gen_list[:20]], 20)
        # ILD = torch.sum(torch.cdist(item2vec1[torch.tensor(pred_list1)],
        #                              item2vec1[torch.tensor(pred_list1)])) / (topk * (topk - 1))
        # 针对每个user计算平均coverage
        if cat_map is not None:
            a = set(gen_list[:5].tolist())
            b = set(gen_list[:10].tolist())
            c = set(gen_list.tolist())
            result[i]["CC@5"] = coverage_for_user(a, cat_map, cat_num)
            result[i]["CC@10"] = coverage_for_user(b, cat_map, cat_num)
            result[i]["CC@20"] = coverage_for_user(c, cat_map, cat_num)
    return result

from sklearn.metrics.pairwise import pairwise_distances
def cal_ILD(gen_vector, topk):
    d = pairwise_distances(gen_vector.numpy(), metric='euclidean')
    ILD =  np.sum(d)/(topk * (topk-1))
    return ILD


def get_coverage(metrics_name, total_set, cat_map, CAT_NUM):

    if metrics_name == "CC@5":
        CC5 = total_set[0]
        if 0 in CC5:
            CC5.remove(0)
        cc = set()
        for item in CC5:
            for cat in cat_map[item]:
                cc.add(cat)
        return len(cc) / CAT_NUM
    elif metrics_name == "CC@10":
        CC10 = total_set[1]
        if 0 in CC10:
            CC10.remove(0)
        cc = set()
        for item in CC10:
            for cat in cat_map[item]:
                cc.add(cat)
        return len(cc) / CAT_NUM
    elif metrics_name == "CC@20":
        CC20 = total_set[2]
        if 0 in CC20:
            CC20.remove(0)
        cc = set()
        for item in CC20:
            for cat in cat_map[item]:
                cc.add(cat)
        return len(cc) / CAT_NUM
    # return 0

def get_metrics_full(metrics_name, total_result):
    if metrics_name == 'recall@5_f':
        recall5 = 0.0
        for i in total_result:
            recall5 += i['recall@5_f']
        return recall5 / len(total_result)
    elif metrics_name == 'mrr@5_f':
        mrr5 = 0.0
        for i in total_result:
            mrr5 += i['mrr@5_f']
        return mrr5 / len(total_result)
    elif metrics_name == 'ndcg@5_f':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@5_f']
        return ndcg5 / len(total_result)
    elif metrics_name == 'recall@10_f':
        recall10 = 0.0
        for i in total_result:
            recall10 += i['recall@10_f']
        return recall10 / len(total_result)
    elif metrics_name == 'mrr@10_f':
        mrr10 = 0.0
        for i in total_result:
            mrr10 += i['mrr@10_f']
        return mrr10 / len(total_result)
    elif metrics_name == 'ndcg@10_f':
        ndcg10 = 0.0
        for i in total_result:
            ndcg10 += i['ndcg@10_f']
        return ndcg10 / len(total_result)
    elif metrics_name == 'recall@20_f':
        recall20 = 0.0
        for i in total_result:
            recall20 += i['recall@20_f']
        return recall20 / len(total_result)
    elif metrics_name == 'mrr@20_f':
        mrr20 = 0.0
        for i in total_result:
            mrr20 += i['mrr@20_f']
        return mrr20 / len(total_result)
    elif metrics_name == 'ndcg@20_f':
        ndcg20 = 0.0
        for i in total_result:
            ndcg20 += i['ndcg@20_f']
        return ndcg20 / len(total_result)
    elif metrics_name == "ILD@5":
        ILD5 = 0.0
        for i in total_result:
            ILD5 += i['ILD@5']
        return ILD5 / len(total_result)
    elif metrics_name == "ILD@10":
        ILD10 = 0.0
        for i in total_result:
            ILD10 += i['ILD@10']
        return ILD10 / len(total_result)
    elif metrics_name == "ILD@20":
        ILD20 = 0.0
        for i in total_result:
            ILD20 += i['ILD@20']
        return ILD20 / len(total_result)
    elif metrics_name == "CC@5":
        CC5 = 0.0
        for i in total_result:
            CC5 += i['CC@5']
        return CC5 / len(total_result)
    elif metrics_name == "CC@10":
        CC10 = 0.0
        for i in total_result:
            CC10 += i['CC@10']
        return CC10 / len(total_result)
    elif metrics_name == "CC@20":
        CC20 = 0.0
        for i in total_result:
            CC20 += i['CC@20']
        return CC20 / len(total_result)
    else:
        raise Exception("参数错误")



######################

def get_metrics(metrics_name,total_result):
    if metrics_name == 'recall@5':
        recall5 = 0.0
        for i in total_result:
            recall5 += i['recall@5']
        return recall5 / len(total_result)
    elif metrics_name == 'mrr@5':
        mrr5 = 0.0
        for i in total_result:
            mrr5 += i['mrr@5']
        return mrr5 / len(total_result)
    elif metrics_name == 'ndcg@5':
        ndcg5 = 0.0
        for i in total_result:
            ndcg5 += i['ndcg@5']
        return ndcg5 / len(total_result)
    elif metrics_name == 'recall@10':
        recall10 = 0.0
        for i in total_result:
            recall10 += i['recall@10']
        return recall10 / len(total_result)
    elif metrics_name == 'mrr@10':
        mrr10 = 0.0
        for i in total_result:
            mrr10 += i['mrr@10']
        return mrr10 / len(total_result)
    elif metrics_name == 'ndcg@10':
        ndcg10 = 0.0
        for i in total_result:
            ndcg10 += i['ndcg@10']
        return ndcg10 / len(total_result)
    elif metrics_name == 'recall@20':
        recall20 = 0.0
        for i in total_result:
            recall20 += i['recall@20']
        return recall20 / len(total_result)
    elif metrics_name == 'mrr@20':
        mrr20 = 0.0
        for i in total_result:
            mrr20 += i['mrr@20']
        return mrr20 / len(total_result)
    elif metrics_name == 'ndcg@20':
        ndcg20 = 0.0
        for i in total_result:
            ndcg20 += i['ndcg@20']
        return ndcg20 / len(total_result)
    else:
        raise Exception("参数错误")



def evaluate_function(output, positives, negatives):  # 增加一个recall@20 去掉auc mrr改成mrr@5 mrr@10 mrr@20
    result = [{} for _ in range(len(output))]
    for i in range(len(output)):
        pos_score = output[i][positives[i]]  # 1 tensor
        neg_scores = output[i][negatives[i]]  # 99 tensor
        success = ((neg_scores - pos_score) < 0).sum()  # tensor
        success = int(success)  # integer
        if 99 - success < 5:
            result[i]['recall@5'] = 1.0
            result[i]['mrr@5'] = 1 / (99 - success + 1)
            result[i]['ndcg@5'] = 1.0 / np.log2(99 - success + 2.0)
        else:
            result[i]['recall@5'] = 0.0
            result[i]['mrr@5'] = 0.0
            result[i]['ndcg@5'] = 0.0
        if 99 - success < 10:
            result[i]['recall@10'] = 1.0
            result[i]['mrr@10'] = 1 / (99 - success + 1)
            result[i]['ndcg@10'] =  1.0 / np.log2(99 - success + 2.0)
        else:
            result[i]['recall@10'] = 0.0
            result[i]['mrr@10'] = 0.0
            result[i]['ndcg@10'] = 0.0
        if 99 - success < 20:
            result[i]['recall@20'] = 1.0
            result[i]['mrr@20'] = 1 / (99 - success + 1)
            result[i]['ndcg@20'] = 1.0 / np.log2(99 - success + 2.0)
        else:
            result[i]['recall@20'] = 0.0
            result[i]['mrr@20'] = 0.0
            result[i]['ndcg@20'] = 0.0
    return result


def cc_at_k(cc, k, CATE_NUM):
    cates = set()
    for i in range(k):
        if i > (len(cc)-1):
            break
        for c in cc[i]: # 每一个
           cates.add(c)
    return len(cates) / CATE_NUM


if __name__ == '__main__':
    pass