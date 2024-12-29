# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train_.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev_.txt'                                    # 验证集
        self.test_path = dataset + '/data/test_.txt'
        self.detect_path = dataset + '/data/input.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 3
        self.batch_size = 16
        self.dropout = 1
        self.pad_size = 32
        self.learning_rate = 5e-5
        self.bert_path = './bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask, token_type_ids=None, return_dict=False)
        out = self.fc(pooled)
        return out
