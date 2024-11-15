# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN_Att'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64


class BILSTMAtt(nn.Module):
    '''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''
    def __init__(self, config_: Config):
        super(BILSTMAtt, self).__init__()

        if config_.embedding_pretrained is not None:
            self.embedding_ = nn.Embedding.from_pretrained(config_.embedding_pretrained, freeze=False)
        else:
            self.embedding_ = nn.Embedding(config_.n_vocab, config_.embed, padding_idx=config_.n_vocab-1)

        self.lstm_ = nn.LSTM(config_.embed, config_.hidden_size, config_.num_layers,
                             bidirectional=True, batch_first=True, dropout=config_.dropout)

        self.tanh1_ = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w_ = nn.Parameter(torch.zeros(config_.hidden_size*2))
        self.tanh2_ = nn.Tanh()
        self.fc1_ = nn.Linear(config_.hidden_size*2, config_.hidden_size2)
        self.fc_ = nn.Linear(config_.hidden_size2, config_.num_classes)

    def forward(self, x_):
        x_, _ = x_
        emb_ = self.embedding_(x_)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        # lstm除了返回输出外，还会返回每一层上的隐藏状态h，其大小为[numLayers*numDirections, batchSize, hiddenSize]
        H_, _ = self.lstm_(emb_)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M_ = self.tanh1_(H_)  # [batch_size, seq_len, hidden_size * num_direction]
        # M = torch.tanh(torch.matmul(H, self.u))
        # 产生注意力权重
        alpha_ = F.softmax(torch.matmul(M_, self.w_), dim=1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        out_ = H_ * alpha_  # [batch_size, seq_len, hidden_size * num_direction]
        out_ = torch.sum(out_, 1)  # [batch_size, hidden_size * num_direction]
        out_ = F.relu(out_)
        out_ = self.fc1_(out_)
        out_ = self.fc_(out_)  # batch_size, numClasses]
        return out_


