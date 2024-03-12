import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from script import ILD_tensor
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
# from torch.nn import MultiheadAttention
import torch.utils.data as tud


class TRIER_PT(nn.Module):
    r"""
    SASRec is the first sequential recommender based on self-attentive mechanism.

    NOTE:
        In the author's implementation, the Point-Wise Feed-Forward Network (PFFN) is implemented
        by CNN with 1x1 kernel. In this implementation, we follows the original BERT implementation
        using Fully Connected Layer to implement the PFFN.
    """

    def __init__(self, n_items, n_layers=1, n_heads=1, hidden_size=64, dropout_prob=0.5, batch_size=256, args=None):
        super(TRIER_PT, self).__init__()

        # load parameters info
        self.n_items = n_items  # note that include padding and EOS and mask
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.inner_size = self.hidden_size * 4
        self.dropout_prob = dropout_prob
        self.hidden_act = 'relu'

        self.lmd = 0.1  # lambda
        self.lmd_sem = 0.1
        self.ssl = args.ssl
        # self.ssl = 'us_x'  # 'us_x'

        self.div = args.div
        # self.kl = args.kl
        # self.reg = True
        self.tau = 1
        self.sim = 'dot'
        self.device = args.device
        self.args = args
        self.inf = torch.tensor([0.0], device=args.device)  # 正无穷 1.0e8
        # define layers and loss
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(100, self.hidden_size)
        trm_encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.n_heads,
                                                    dim_feedforward=self.inner_size, dropout=self.dropout_prob,
                                                    activation=self.hidden_act)
        self.trm_encoder = nn.TransformerEncoder(trm_encoder_layer, self.n_layers)

        # note that d_model%nhead=0 and the final layer of TransformerEncoderLayer is dropout
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)


        self.batch_size = batch_size
        self.mask = self.mask_correlated_samples(self.batch_size)

        self.nce_fct = nn.CrossEntropyLoss(reduction='mean')
        self.KL_loss = nn.KLDivLoss() # reduction="mean"
        # parameters initialization
        # self.apply(self._init_weights)


    def forward_RT(self, seq_token, input_reverse_ids, reverse_model=None, use_decoder=False):
        # 通过beam search的方式生成多个序列，在原始序列的左边补齐，
        beam_width = self.args.bw
        batch_size = input_reverse_ids.shape[0]
        input_length = (input_reverse_ids > 0).sum(-1)
        # print(input_length.device, input_reverse_ids.device)
        new_batch_size = batch_size * beam_width
        # 生成整个input的Embedding, RT 模型学习的是反向预测，所以H left为正常用于反向预测的

        H_left = reverse_model.forward(input_reverse_ids, input_length)  # B x H
        next_probabilities = torch.matmul(H_left, reverse_model.item_embedding.weight.T).log_softmax(-1)  # BxH * HxV = BxV
        probabilities, idx = next_probabilities.topk(k=beam_width, axis=-1)  # 返回值和索引  # B x W

        input_reverse_ids = input_reverse_ids.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)  # (B x W) X max_len
        seq_token = seq_token.repeat((beam_width, 1, 1)).transpose(0, 1).flatten(end_dim=-2)  # (B x W) X max_len
        input_length = input_length.repeat((beam_width, 1)).transpose(0, 1).reshape(new_batch_size)
        gen_token = idx.view(-1, 1).squeeze(-1)  # (B x W) x 1
        # 一个 BxW的，一个B维的
        # input_reverse_ids, rec_list, _, probabilities = reverse_model.beam_search_gen(input_reverse_ids, probabilities, gen_token, input_length)
        input_reverse_ids, rec_list, _, probabilities = reverse_model.k_select_1(input_reverse_ids, probabilities,
                                                                                 gen_token, input_length, predictions=self.args.lm)
        gen_items = torch.flip(rec_list, dims=[1])
        seq_token = torch.roll(seq_token, shifts=gen_items.shape[1])
        seq_token[torch.arange(new_batch_size), : gen_items.shape[1]] = gen_items
        # 生成每个beam的Embedding,作为intent, # 此处应该用正向预测的model
        if use_decoder:
            F = self.forward_decoder(seq_token, None).view(batch_size, beam_width, -1)  # 注意这里的seq token 需要修改为增广部分的内容
        else:
            F = self.forward(seq_token, input_length).view(batch_size, beam_width, -1)  # 注意这里的seq token 需要修改为增广部分的内容

        # reverse_model.item_embedding(torch.stack(rec_list,dim=1).view(4,3,5)).permute(2,0,1,3) 和下面相同
        # rec_emb = reverse_model.item_embedding(rec_list).view(-1, batch_size, beam_width, self.hidden_size)
        # 此处的probability是log的累加和，即生成该序列的概率log值
        return F, probabilities

    def forward(self, input_session_ids, item_seq_len):  # 这里的input_session_ids不包含start token，在此基础上要保证input_session_ids不为空序列

        input_emb =self.embedding(input_session_ids)

        padding_mask = (input_session_ids == 0)  # 要mask的位置为True
        src_mask = (1-torch.tril(torch.ones(input_emb.shape[0], input_emb.shape[0], device=padding_mask.device))).bool()  # [seq_len, seq_len]
        output = self.trm_encoder(input_emb, mask=src_mask, src_key_padding_mask=padding_mask, )
        output = output.permute(1, 0, 2)  # [batch_size, seq_len, emb_size]
        output = self.gather_indexes(output, item_seq_len - 1)  # [batch_size, emb_size]，注意只计算最后一个时间步（即要预测的下一个物品）的损失（这里把最后一个时间步的输出也看成是整个序列的表示）
        return output
        # try:
        #     # print(item_seq_len)
        #     output = self.gather_indexes(output, item_seq_len - 1)  # [batch_size, emb_size]，注意只计算最后一个时间步（即要预测的下一个物品）的损失（这里把最后一个时间步的输出也看成是整个序列的表示）
        #     return output
        # except:
        #     # print(item_seq_len)
        #     item_seq_len = torch.clamp(item_seq_len, min=1)
        #     print(item_seq_len)
        #     output =self.gather_indexes(output, item_seq_len - 1)
        #     return output

    def train_forward(self, input_session_ids, sem_aug_input_session_ids, input_reverse_ids, rt_model, item2vec):
        # 所谓sem_aug_input_session_ids就是与input_session_ids有着相同的要预测的下一个物品的另一个用户的序列
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len) # encoder模块输出，最后一个
        div_loss, nce_loss = 0, 0

        # B x K X H

        # weight = torch.ones_like(weight).softmax(-1).detach()
        if self.div:
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)  # 向左增广得到的hidden state
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            output_logit, output_logit_greedy, output_token, output_token_greedy = \
                self.generate_by_score(input_session_ids, output, F, weight) #
            div_loss = self.diversity_loss(output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec)


        if self.ssl == 'us_x':
            nce_loss = self.contrastive_loss(input_session_ids, sem_aug_input_session_ids, item_seq_len)

        return output, nce_loss, div_loss#, kl_loss  # attn_output

        # return attn_output, nce_loss, div_loss, kl_loss

    def generate_by_score(self, input_session_ids, output, F, attention_weght, test=False):

        batch_size = input_session_ids.shape[0]
        device = input_session_ids.device
        output_token, output_token_greedy = [], []  # 记录每一步向右生成的token, || 记录不使用memory生成的token
        output_logit, output_logit_greedy = [], []  # 记录每一步向右生成的logit || 记录不考虑多样性向右生成的logit
        index_dim0 = torch.arange(batch_size)  # [B], range(0,batch_size)
        tgt_seq_length = self.args.k if not test else 20
        # tgt_seq_length = random.randint(2,20) if not test else 20
        # tgt_seq_length =
        mask = torch.ones(batch_size, self.n_items, device=device, requires_grad=False)  # B x V
        mask[:, 0] = self.inf
        mask_greedy = mask.clone()
        H_input = output.clone()
        logit = torch.matmul(H_input, self.item_embedding.weight.T) # B x V
        rel_score = logit.softmax(-1)
        rel_score = rel_score * mask.clone()
        for cnt in range(tgt_seq_length):
            if cnt == 0: # 推荐的第一个item只根据准确性评分
                top1 = rel_score.argmax(-1)
                greedy_top1 = top1.clone()
                score = rel_score.clone()
            else:
                # rel_score, score = self.calculate_score(rel_score, top1.unsqueeze(-1), F, attention_weght)
                score = self.calculate_score(rel_score, output_token, F, attention_weght)
                top1 = (score * mask.clone()).argmax(-1)
                greedy_top1 = (rel_score * mask_greedy.clone()).argmax(-1)
            mask[index_dim0, top1] = self.inf  # 将已生成位置的概率设置为负无穷
            mask_greedy[index_dim0, greedy_top1] = self.inf
            output_logit.append(score) # logit
            output_token.append(top1)
            output_logit_greedy.append(rel_score)  # logit_greedy
            output_token_greedy.append(greedy_top1)

        output_token = torch.stack(output_token, dim=1)
        output_token_greedy = torch.stack(output_token_greedy, dim=1)
        if test:
            return output_token # test_out
        return output_logit, output_logit_greedy, output_token, output_token_greedy



    def calculate_score(self, rel_score, output_token, F, attention_weght):
        # 要从第二个step开始，因为第一个step没有已推荐列表，没法计算div score
        # trade off parameter
        lamb = self.args.lamb
        # k = self.args.bw
        # diversity score
        P_va = (torch.matmul(F,  self.item_embedding.weight.T)*10).softmax(-1)   # 乘10  # B x K X V
        P_a_u = attention_weght + 1e-24 # B x 1 X K  # 可能存在0
        # output_token = torch.stack(output_token, dim=1) # Bx1
        # H_y  = self.item_embedding(output_token) # Bx1XH
        output_token = torch.stack(output_token, dim=1)
        # print("#############################")
        # print(output_token)
        a,b = output_token.shape

        H_y_input = torch.cat((output_token, torch.zeros((a, self.args.mml - b), device=self.device)), dim=1)
        # 此处应该用一个新的encoder模块
        H_y  = self.forward(H_y_input.long(), (H_y_input > 0).sum(-1)).unsqueeze(1)

        w_i = torch.matmul(F, H_y.transpose(1,2)).transpose(1,2).softmax(-1) # B x 1 x K
        temp = P_a_u * w_i
        W_Ra = 1 - temp/(temp.sum(-1).unsqueeze(1) + 1e-24) # B x 1 X K
        div_score = torch.matmul(W_Ra, P_va).squeeze(1)#.softmax(-1) # B x 1 x V
        # summary
        # score = lamb * rel_score + (1 - lamb) * div_score
        score = lamb * div_score + (1 - lamb) * rel_score
        return score


    def diversity_loss(self, output_logit, output_logit_greedy, output_token, output_token_greedy, item2vec):

        output_logit = torch.stack(output_logit, dim=1)#.log()#.log_softmax(-1)  # B x L x V
        output_logit_greedy = torch.stack(output_logit_greedy, dim=1)#.log()#.log_softmax(-1)
        pre_logit = output_logit.gather(-1, output_token.unsqueeze(-1)).squeeze(-1)
        pre_logit_greedy = output_logit_greedy.gather(-1, output_token_greedy.unsqueeze(-1)).squeeze(-1)
        item2vec = item2vec.to(self.device)
        ILD_ture = ILD_tensor(output_token, item2vec)  # .to(device)
        ILD_greedy = ILD_tensor(output_token_greedy, item2vec)  # .to(device)
        # 待检查，P_rel和P的大小关系是否一定是前面大，后面小
        P_rel = pre_logit_greedy.log().sum(-1)
        P = pre_logit.log().sum(-1)
        # div_loss = (ILD_greedy - ILD_ture) *((1 / (1 + torch.exp(P_rel - P))).log())
        div_loss = -1 * (ILD_greedy - ILD_ture) * (1 + torch.exp((P_rel - P)/10)).log()
        # print(pre_logit, P)
        # div_loss = 1 / (1 + torch.exp(P_rel - P))
        div_loss = div_loss.mean()

        return div_loss


    def contrastive_loss(self, input_session_ids, sem_aug_input_session_ids, item_seq_len, use_decoder=False):
        if not use_decoder:
            aug_output = self.forward(input_session_ids, item_seq_len)
            sem_aug_item_seq_len = (sem_aug_input_session_ids > 0).sum(-1)
            sem_aug_output = self.forward(sem_aug_input_session_ids, sem_aug_item_seq_len)
        else:
            aug_output = self.forward_decoder(input_session_ids, None)
            sem_aug_output = self.forward_decoder(sem_aug_input_session_ids, None)
        sem_nce_logits, sem_nce_labels = self.info_nce(aug_output, sem_aug_output, temp=self.tau,
                                                       batch_size=item_seq_len.shape[0], sim=self.sim)
        nce_loss = self.lmd_sem * self.nce_fct(sem_nce_logits, sem_nce_labels)

        return nce_loss


    def rec_loss(self, output, targets, nce_loss, div_loss):
        output = torch.matmul(output, self.item_embedding.weight.T)  # [batch_size, item_num]
        targets = targets.unsqueeze(-1)  # [batch_size, 1]
        rec_loss = -output.log_softmax(dim=-1).gather(dim=-1, index=targets).squeeze(-1)  # [batch_size]
        main_loss = rec_loss.mean()
        loss = main_loss + nce_loss + div_loss #+ kl_loss
        return loss, main_loss


    def replace(self, seq_token, index_dim0, input_length, gen_token):
        seq_token1 = seq_token.clone()
        seq_token1[index_dim0, input_length] = gen_token
        return seq_token1


    def test_forward(self, input_session_ids, input_reverse_ids=None, rt_model=None,
                      step_by_step=False):
        item_seq_len = (input_session_ids > 0).sum(-1)  # [batch_size]
        output = self.forward(input_session_ids, item_seq_len)

        if step_by_step:
            F, probabilities = self.forward_RT(input_session_ids, input_reverse_ids, rt_model)
            # attn_output_weight = attn_output_weight.softmax(-1)
            # B x1 X H, Bx1xK
            weight = probabilities.softmax(-1).unsqueeze(1).detach()
            # torch.save(probabilities.softmax(-1), "weight.pt")
            # weight = torch.ones_like(weight).softmax(-1).detach()
            output_token = self.generate_by_score(input_session_ids, output, F, weight, test=True)
            return output, output_token
        else:
            return output


    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

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

    def embedding(self, input_session_ids):
        position_ids = torch.arange(input_session_ids.size(1), dtype=torch.long, device=input_session_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_session_ids)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(input_session_ids)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        input_emb = input_emb.permute(1, 0, 2)  # [seq_len, batch_size, emb_size]
        return input_emb
