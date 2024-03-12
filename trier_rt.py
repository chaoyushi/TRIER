import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import torch.utils.data as tud

from pytorch_beam_search import autoregressive

class TRIER_RT(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, batch_size=256, args=None):
        super(TRIER_RT, self).__init__()

        # load parameters info
        self.n_items = n_items  # note that include padding and EOS and mask
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = self.hidden_size*4
        self.dropout_prob = dropout_prob
        self.hidden_act = 'relu'
        
        self.lmd = 0.1   #lambda
        self.lmd_sem = 0.1
        self.ssl = 'us_x' # 'us_x'
        # self.div = True
        self.reg = args.reg
        self.tau = 1
        self.sim = 'dot'
        self.args = args
        self.inf = torch.tensor([1.0e8], device=args.device)  # 正无穷
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        trm_encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                    dim_feedforward=self.inner_size, dropout=self.dropout_prob, activation=self.hidden_act)
        self.trm_encoder = nn.TransformerEncoder(trm_encoder_layer, self.n_layers)
        # note that d_model%nhead=0 and the final layer of TransformerEncoderLayer is dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)

        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')

        # parameters initialization
        #self.apply(self._init_weights)

    # def _init_weights(self, module):
    #     """ Initialize the weights """
    #     if isinstance(module, (nn.Linear, nn.Embedding)):
    #         # Slightly different from the TF version which uses truncated_normal for initialization
    #         # cf https://github.com/pytorch/pytorch/pull/5617
    #         module.weight.data.normal_(mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         module.bias.data.zero_()
    #         module.weight.data.fill_(1.0)
    #     if isinstance(module, nn.Linear) and module.bias is not None:
    #         module.bias.data.zero_()

    def gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)  # 屏蔽掉与自己的sim
        for i in range(batch_size):
            mask[i, batch_size + i] = 0  # 屏蔽掉与对应的另一个变换的sim
            mask[batch_size + i, i] = 0
        return mask

    def info_nce(self, z_i, z_j, temp, batch_size, sim='dot'):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017),
        we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        （对应的另一个变换作为正例）
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)  # [2*batch_size, emb_size]

        if sim == 'cos':
            sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
        elif sim == 'dot':
            sim = torch.mm(z, z.T) / temp  # [2*batch_size, 2*batch_size]
        # 这两个sim是相同的
        sim_i_j = torch.diag(sim, batch_size)  # [batch_size]
        sim_j_i = torch.diag(sim, -batch_size)  # [batch_size]
        # 与对应的另一个变换的sim，[2*batch_size, 1]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
            self.mask = self.mask_correlated_samples(batch_size)
        mask = self.mask
        # 与负例的sim，[2*batch_size, 2*batch_size-2]
        negative_samples = sim[mask].reshape(N, -1)
        # [2*batch_size]，之所以全是0是因为positive_samples在logits里排在第一列
        labels = torch.zeros(N).to(positive_samples.device).long()
        # [2*batch_size, 2*batch_size-1]（第一列对应正例，其它列对应负例）
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        return logits, labels

    def forward(self, input_session_ids, item_seq_len):  # 这里的input_session_ids不包含start token，在此基础上要保证input_session_ids不为空序列
        position_ids = torch.arange(input_session_ids.size(1), dtype=torch.long, device=input_session_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(input_session_ids)
        output = item_emb + position_embedding
        output = self.LayerNorm(output)
        output = self.dropout(output)

        output = output.permute(1, 0, 2)  # [seq_len, batch_size, emb_size]
        padding_mask = (input_session_ids == 0)  # 要mask的位置为True
        # src_mask = (1-torch.tril(torch.ones(output.shape[0], output.shape[0], device=padding_mask.device))).bool()  # [seq_len, seq_len]
        output = self.trm_encoder(output, src_key_padding_mask=padding_mask).permute(1, 0, 2) # [batch_size, seq_len, emb_size]
        # 注意只计算最后一个时间步（即要预测的下一个物品）的损失（这里把最后一个时间步的输出也看成是整个序列的表示）
        output = self.gather_indexes(output, item_seq_len - 1)  # [batch_size, emb_size]，
        return output

    def train_forward(self, input_session_ids, sem_aug_input_session_ids):  # 所谓sem_aug_input_session_ids就是与input_session_ids有着相同的要预测的下一个物品的另一个用户的序列
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len)
        div_loss, nce_loss = 0, 0
        dis_reg, me_reg = 0, 0

        # us_x
        if self.ssl == 'us_x':
            aug_output = self.forward(input_session_ids, item_seq_len)
            sem_aug_item_seq_len = (sem_aug_input_session_ids > 0).sum(-1)
            sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_item_seq_len)
            sem_nce_logits, sem_nce_labels = self.info_nce(aug_output, sem_aug_output, temp=self.tau, batch_size=item_seq_len.shape[0], sim=self.sim)
            nce_loss += self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)

        if self.reg:
            dis_reg, me_reg = self.generate_step_by_step(input_session_ids, output)
        return output, nce_loss, dis_reg, me_reg

    def generate_step_by_step(self, input_session_ids, output, test=False):

        output_logit = []
        beam_width = self.args.bw
        batch_size = input_session_ids.shape[0]
        logit = torch.matmul(output, self.item_embedding.weight.T)
        probabilities, idx = logit.log_softmax(-1).topk(k=beam_width, axis=-1)  # 返回值和索引  # B x W
        output_logit.append(logit.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2))

        input_length = (input_session_ids > 0).sum(-1)
        seq_token = input_session_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)  # (B x W) X max_len
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1) \
            .reshape(batch_size * beam_width)
        gen_token = idx.view(-1, 1).squeeze(-1)  # (B x W) x 1
        # beam search step 2-n
        # _, rec_list, _, probabilities = self.beam_search_gen(seq_token, probabilities, gen_token, input_length,
        #                     output_logit=output_logit)
        _, rec_list, _, probabilities = self.k_select_1(seq_token, probabilities, gen_token, input_length,
                                                        output_logit=output_logit)

        # 生成每个beam的Embedding,作为intent, B x W X H
        if test:
            return rec_list
        retro_emb = self.item_embedding(rec_list).view(-1, batch_size, beam_width, self.hidden_size)
        # 计算 loss
        Dis_reg = torch.matmul(retro_emb, retro_emb.permute(0, 1, 3, 2)).view(batch_size, -1).sum(-1)
        Dis_reg = Dis_reg.mean()
        temp = probabilities.exp()
        ME_reg = 0.1 * (temp / (temp.sum(-1).unsqueeze(1)) * probabilities).sum(-1).mean()
        return Dis_reg, ME_reg

    def k_select_1(self, seq_token, probabilities, gen_token, input_length, output_logit=None, predictions=None):

        device = seq_token.device
        rec_list = []
        rec_list.append(gen_token)
        beam_width = self.args.bw
        new_batch_size = seq_token.shape[0]  # batch_size * beam_width
        batch_size = int(new_batch_size / beam_width)
        mask = torch.ones(new_batch_size, self.n_items, device=device, requires_grad=False)
        index_dim0 = torch.arange(new_batch_size, device=device)
        # mask = mask.reshape()
        mask[index_dim0, gen_token] = self.inf
        # 一个 BxW的，一个B维的
        seq_token[index_dim0, input_length] = gen_token
        input_length = input_length + 1
        # 循环，通过 增广+beam search的方式，生成新的item list。
        step = self.args.k if predictions == None else predictions
        # print("########################################",step, "########################################")
        probabilities = probabilities.unsqueeze(-1)
        for _ in range(step - 1):
            dataset = tud.TensorDataset(seq_token, input_length)  # \
            # if reverse_seq_pos == None else tud.TensorDataset(seq_token, reverse_seq_pos) # RT,  PT
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            logits = []
            # 循环新的batch data
            for (x, x_len) in iter(loader):
                # if reverse_seq_pos == None:  #RT 模型是其本身
                logits.append(torch.matmul(self.forward(x, x_len), self.item_embedding.weight.T))
                # else: # PT,模型是导入的，同时token emb也变了
                #     logits.append(torch.matmul(reverse_model.forward(x, x_pos), token_emb))
            logits = torch.cat(logits, dim=0)  # output
            if output_logit != None:
                output_logit.append(logits)
            next_probabilities = logits.log_softmax(-1).view((-1, beam_width, self.n_items))
            probabilities = (probabilities + next_probabilities) # .flatten(start_dim=1)  # 广播，BxWx1 + BxWXV
            # 保证不重复性
            # print(probabilities.device, mask.device)
            probabilities = probabilities * (mask.clone().view(batch_size, beam_width,  -1))# (mask.clone().view(batch_size, -1))
            probabilities, idx = probabilities.topk(k=1, axis=-1)
            gen_token = idx.flatten()
            # gen_token = torch.remainder(idx, self.n_items).flatten()  # 模除，映射到对应的item id
            mask[index_dim0, gen_token] = self.inf
            rec_list.append(gen_token)
            seq_token[index_dim0, input_length] = gen_token
            input_length = input_length + 1
        rec_list = torch.stack(rec_list, dim=1)
        return seq_token, rec_list, output_logit, probabilities.squeeze(-1)


    def test_forward(self, input_session_ids):
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len)

        return output

    #'''
    def rec_loss(self, output, targets, nce_loss, dis_reg, me_reg):
        output = torch.matmul(output, self.item_embedding.weight.T)  # [batch_size, item_num]
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)  # [batch_size]
        main_loss = rec_loss.mean()
        loss = main_loss + nce_loss + dis_reg + me_reg
        return loss


    def beam_search_gen(self, seq_token, probabilities, gen_token, input_length, output_logit=None, predictions=None):
        device = seq_token.device
        rec_list = torch.cat((gen_token.view(-1,1),), dim=-1)
        # rec_list.append(gen_token) # 需要使用rec list回溯
        beam_width = self.args.bw
        new_batch_size = seq_token.shape[0]  #  batch_size * beam_width
        batch_size = int(new_batch_size / beam_width)
        index_dim0 = torch.arange(new_batch_size, device=device)
        src_mask = torch.ones(new_batch_size, self.n_items, device=device, requires_grad=False)
        mask = src_mask.clone()
        mask[index_dim0, gen_token] = self.inf
        modified_index = torch.stack([input_length])
        seq_token[index_dim0, input_length] = gen_token
        input_length = input_length + 1
        modified_index = torch.cat((modified_index, input_length.unsqueeze(0)))

        # 循环，通过 增广+beam search的方式，生成新的item list。
        step = self.args.k if predictions == None else predictions
        for _ in range(step - 1):
            dataset = tud.TensorDataset(seq_token, input_length) #\
                # if reverse_seq_pos == None else tud.TensorDataset(seq_token, reverse_seq_pos) # RT,  PT
            loader = tud.DataLoader(dataset, batch_size=batch_size)
            logits = []
            # 循环新的batch data
            for (x, x_len) in iter(loader):
                logits.append(torch.matmul(self.forward(x, x_len), self.item_embedding.weight.T))
            logits = torch.cat(logits, dim=0)  # output
            if output_logit != None:
                output_logit.append(logits)
            next_probabilities = logits.log_softmax(-1).view((-1, beam_width, self.n_items))
            probabilities = (probabilities.unsqueeze(-1) + next_probabilities).flatten(start_dim=1)  # 广播，BxWx1 + BxWXV
            # 保证不重复性
            probabilities = probabilities * (mask.clone().view(batch_size,-1))
            probabilities, idx = probabilities.topk(k=beam_width, axis=-1)

            gen_token = torch.remainder(idx, self.n_items).flatten().unsqueeze(-1)  # 模除，映射到对应的item id

            best_candidates = (idx / self.n_items).long()
            best_candidates += torch.arange(batch_size, device=device).unsqueeze(-1) * beam_width
            rec_list = rec_list[best_candidates].flatten(end_dim = -2)
            rec_list = torch.cat((rec_list, gen_token), dim=1)
            # X = torch.cat((X, next_chars), axis = 1)
            mask = src_mask.clone()
            mask[index_dim0, rec_list.T] = self.inf
            seq_token[index_dim0, modified_index] = rec_list.T
            input_length = input_length + 1
            modified_index = torch.cat((modified_index, input_length.unsqueeze(0)))

        return seq_token, rec_list, output_logit, probabilities

    '''
    def decompose(self, z_i, z_j, origin_z, batch_size):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)

        # pairwise l2 distace
        sim = torch.cdist(z, z, p=2)

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        alignment = positive_samples.mean()

        # pairwise l2 distace
        sim = torch.cdist(origin_z, origin_z, p=2)
        mask = torch.ones((batch_size, batch_size), dtype=bool)
        mask = mask.fill_diagonal_(0)
        negative_samples = sim[mask].reshape(batch_size, -1)
        uniformity = torch.log(torch.exp(-2 * negative_samples).mean())

        return alignment, uniformity
    '''
    #'''
    '''  
    def rec_loss(self, output, targets, negatives, nce_loss):        
        target_embs = self.item_embedding(targets)  # [batch_size, emb_size]
        negative_embs = self.item_embedding(negatives)  # [batch_size, emb_size]
        pos_score = (output*target_embs).sum(-1)  # [batch_size]
        neg_score = (output*negative_embs).sum(-1)  # [batch_size]
        rec_loss = -(torch.log(torch.sigmoid(pos_score) + 1e-24) + torch.log(1 - torch.sigmoid(neg_score) + 1e-24))  # [batch_size]
        loss = rec_loss.mean()
        loss += nce_loss
        return loss
    '''
