# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import os

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, default='BILSTMAtt', help='choose a model: BILSTMAtt')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')  # word==False，对于中文则是文字、标点级
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # BILSTMAtt

    from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)

    if not os.path.exists(os.path.dirname(config.save_path)):
        os.makedirs(os.path.dirname(config.save_path), exist_ok=True)

    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path, exist_ok=True)

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)  # 为所有的GPU设置生成随机数种子
    # 用来调整PyTorch中使用的CuDNN（NVIDIA的CUDA深度学习加速库）的行为，保证确定性，即每次结果一样
    # 当在CuDNN后端运行时，需要进一步设置torch.backends.cudnn.benckmark = False，来关闭CuDNN的自动优化，保证每次运行结果一致
    # 若要保证效率，则设置torch.backends.cudnn.deterministic = False、torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False：允许CuDNN使用非确定性的算法来增加训练速度，对于调试和需要结果完全可复现的场景不适用，但对于追求最高训练效率的情况是有益的
    # torch.backends.cudnn.benchmark = True：开启了一种性能优化模式，CuDNN会在首次运行时对卷积层等操作的不同实现进行 benchmarking（性能测试），
    # 并选择最快的一个实现方案，即在每个卷积操作中自动寻找最适合当前配置的高效算法，从而提高计算速度
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    # 数据迭代器构建
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 模型构建
    config.n_vocab = len(vocab)
    model = x.BILSTMAtt(config).to(config.device)
    init_network(model)
    print(model.parameters)

    train(config, model, train_iter, dev_iter, test_iter)
