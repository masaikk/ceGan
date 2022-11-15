#
# Copyright (C), 2018-下午9:24
# FileName: CeGAN_D_sin.py
# Author:   b8313
# Date:     下午9:24 下午9:24
# Description:
#


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.discriminator import CNNDiscriminator

dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class CeGAN_D(nn.Module):
    def __init__(self, embed_dim, vocab_size, padding_idx, max_seq_len, num_rep, gpu=False, dropout=0.2):
        super(CeGAN_D, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.max_seq_len = max_seq_len
        self.num_rep = num_rep
        self.gpu = gpu
        self.sen_dropout = dropout
        self.vec_dropout = dropout

        self.feature_dim = sum(dis_num_filters)
        self.emb_dim_single = int(embed_dim / num_rep)
        # this is  copy from rel-gan

        self.vec_dis = CNNDiscriminator(self.embed_dim, self.vocab_size, dis_num_filters, dis_num_filters,
                                        self.padding_idx, self.gpu, self.vec_dropout)
        # This discriminator is for the word embedding in rel-gan

        self.sen_dis = CNNDiscriminator(self.embed_dim, self.vocab_size, dis_filter_sizes, dis_num_filters,
                                        self.padding_idx, self.gpu, self.sen_dropout)
        # This discriminator is for the sentence in leak-gan



        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 100)
        self.out2logits = nn.Linear(100, 1)
        self.dropout = nn.Dropout(dropout)
        # These for vec dis

        self.vec_dis.init_params()
        self.sen_dis.init_params()
        # for these two discriminators

    def forward(self, inp):
        """
        to forward
        :param inp:
        :return:
        """

        return inp


    def forward_sen_dis(self,inp):
        return inp






























    def forward_vec_dis(self,inp):
        """
        Get logits of discriminator
        :param inp: batch_size * seq_len * vocab_size
        :return logits: [batch_size * num_rep] (1-D tensor)
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim

        cons = [F.relu(conv(emb)) for conv in self.convs]  # [batch_size * num_filter * (seq_len-k_h+1) * num_rep]
        pools = [F.max_pool2d(con, (con.size(2), 1)).squeeze(2) for con in cons]  # [batch_size * num_filter * num_rep]
        pred = torch.cat(pools, 1)
        pred = pred.permute(0, 2, 1).contiguous().view(-1, self.feature_dim)  # (batch_size * num_rep) * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        pred = self.feature2out(self.dropout(pred))
        logits = self.out2logits(pred).squeeze(1)  # [batch_size * num_rep]

        return logits
