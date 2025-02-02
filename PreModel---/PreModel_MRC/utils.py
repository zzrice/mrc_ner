#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""utils"""
import json
import os
from pathlib import Path
import shutil
import logging

import torch
import torch.nn as nn
import torch.nn.init as init

ENTITY_TYPE = ['疾病和诊断', '影像检查', '实验室检验', '手术', '药物', '解剖部位']
IO2STR = {
    'DIS': '疾病和诊断',
    'SCR': '影像检查',
    'LAB': '实验室检验',
    'OPE': '手术',
    'MED': '药物',
    'POS': '解剖部位'
}
EN2QUERY = {
    '疾病和诊断': '找出疾病和诊断,例如癌症,病变,炎症,增生,肿瘤',
    '影像检查': '找出影像检查,例如CT,CR,MRI,彩超,X光,造影',
    '实验室检验': '找出实验室检验,例如血型,比率,酶,蛋白,PH',
    '手术': '找出手术,例如切除,检查,根治,电切,穿刺,移植',
    '药物': '找出药物,例如胶囊',
    '解剖部位': '找出解剖部位,例如胃,肠,肝,肺,胸,臀'
}


class Params:

    def __init__(self, ex_index=1):
        """
        Args:
            ex_index (int): 实验名称索引  # 实验名称索引
        """
        # 根路径
        self.root_path = Path(os.path.abspath(os.path.dirname(__file__)))
        self.data_dir = self.root_path / 'data'
        self.params_path = self.root_path / f'experiments/ex{ex_index}'
        self.bert_model_dir = self.root_path.parent.parent / 'nezha-large'
        self.model_dir = self.root_path / f'model/ex{ex_index}'

        # 读取保存的data
        self.data_cache = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_gpu = torch.cuda.device_count()

        self.train_batch_size = 32  # 增加训练batch size
        self.val_batch_size = 128  # 增加验证batch size
        self.test_batch_size = 256  # 保持测试batch size不变

        # patience策略
        self.patience = 0.1  # 提升阈值
        self.patience_num = 3  # 耐心次数
        self.min_epoch_num = 3  # 最小epoch数

        # 标签列表
        self.tag_list = ENTITY_TYPE  # 标签列表
        self.max_seq_length = 128  # 最大序列长度

        self.fusion_layers = 4  # 融合层数
        self.dropout = 0.3  # dropout率
        self.weight_decay_rate = 0.1  # 权重衰减率
        self.fin_tuning_lr = 3e-5  # 略微提高微调学习率
        self.downstream_lr = 2e-4  # 略微提高下游任务学习率
        # 梯度截断
        self.clip_grad = 2  # 梯度截断值
        self.warmup_prop = 0.1  # warmup比例
        self.gradient_accumulation_steps = 2  # 梯度累积步数

        # SMART对抗训练参数
        self.use_smart = True  # 是否使用SMART对抗训练
        self.smart_step_size = 1e-3  # SMART步长
        self.smart_noise_var = 1e-5  # SMART噪声方差
        self.smart_num_steps = 1  # SMART对抗步数
        self.smart_loss_weight = 1.0  # SMART损失权重

    def get(self):
        """以字典形式访问Params实例的所有参数
        
        返回:
            dict: 包含所有参数的字典，可通过类似字典的方式访问参数值
            示例: params.get()['learning_rate']
        """
        return self.__dict__

    def load(self, json_path):
        """从JSON文件加载参数并更新当前实例
        
        参数:
            json_path (str): JSON文件路径，包含要加载的参数
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        """将当前参数保存到JSON文件
        
        仅保存基本数据类型（str, int, float, bool）的参数值
        
        参数:
            json_path (str): 保存参数的JSON文件路径
        """
        params = {}
        with open(json_path, 'w') as f:
            for k, v in self.__dict__.items():
                if isinstance(v, (str, int, float, bool)):
                    params[k] = v
            json.dump(params, f, indent=4)


class RunningAverage:
    """用于维护数值的运行平均值
    
    示例:
        loss_avg = RunningAverage()
        loss_avg.update(2)
        loss_avg.update(4)
        loss_avg()  # 返回3.0
    """

    def __init__(self):
        """初始化运行平均值计算器"""
        self.steps = 0  # 更新次数
        self.total = 0  # 累计值

    def update(self, val):
        """更新运行平均值
        
        参数:
            val (float): 要加入平均值计算的新值
        """
        self.total += val
        self.steps += 1

    def __call__(self):
        """获取当前运行平均值
        
        返回:
            float: 当前运行平均值
        """
        return self.total / float(self.steps)


def save_checkpoint(state, is_best, checkpoint):
    """保存模型和训练参数到指定路径
    
    如果is_best为True，同时保存最佳模型副本
    
    参数:
        state (dict): 包含模型状态等的字典，可能包含epoch、optimizer等信息
        is_best (bool): 是否为当前最佳模型
        checkpoint (str): 保存checkpoint的目录路径
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint目录不存在，正在创建目录: {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, optimizer=True):
    """从指定路径加载模型
    
    如果optimizer为True，同时加载优化器状态
    
    参数:
        checkpoint (str): 要加载的checkpoint文件路径
        optimizer (bool): 是否加载优化器状态
        
    返回:
        tuple: (模型, 优化器) 或仅返回模型
    """
    if not os.path.exists(checkpoint):
        raise ValueError("文件不存在: {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']


def set_logger(save, log_path=None):
    """设置日志记录器，将日志输出到终端和文件
    
    参数:
        save (bool): 是否保存日志到文件
        log_path (str): 日志文件保存路径
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def initial_parameter(net, initial_method=None):
    """初始化PyTorch模型参数
    
    支持多种初始化方法：
        - xavier_uniform
        - xavier_normal (默认)
        - kaiming_normal/msra
        - kaiming_uniform
        - orthogonal
        - sparse
        - normal
        - uniform
    
    参数:
        net: PyTorch模型或模型列表
        initial_method (str): 初始化方法名称
    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        """权重初始化函数，处理不同类型的层"""
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv3d):
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)
                else:
                    init.normal_(w.data)
        elif m is not None and hasattr(m, 'weight') and hasattr(m.weight, "requires_grad"):
            if len(m.weight.size()) > 1:
                init_method(m.weight.data)
            else:
                init.normal_(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)
                    else:
                        init.normal_(w.data)

    if isinstance(net, list):
        for n in net:
            n.apply(weights_init)
    else:
        net.apply(weights_init)


class SMARTLoss(nn.Module):
    """SMART Loss的实现
    
    参考论文: SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization
    """
    def __init__(self, eval_fn, loss_fn, num_steps=1, step_size=1e-3, epsilon=1e-6, noise_var=1e-5):
        """初始化SMART Loss
        
        Args:
            eval_fn: 评估函数，接收扰动后的embeddings并返回模型输出
            loss_fn: 损失函数，计算预测值和目标值之间的损失
            num_steps: 对抗训练的步数
            step_size: 对抗扰动的步长
            epsilon: 用于数值稳定性的小常数
            noise_var: 初始随机扰动的方差
        """
        super(SMARTLoss, self).__init__()
        self.eval_fn = eval_fn
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon
        self.noise_var = noise_var

    def forward(self, embed, pred, targ):
        """计算SMART Loss
        
        Args:
            embed: 原始embeddings
            pred: 原始预测结果
            targ: 目标值
            
        Returns:
            torch.Tensor: SMART Loss值
        """
        # 初始化对抗扰动
        noise = torch.randn_like(embed) * self.noise_var
        noise.requires_grad_()
        
        # 对抗训练
        for _ in range(self.num_steps):
            # 前向传播
            adv_pred = self.eval_fn(embed + noise)
            adv_loss = self.loss_fn(adv_pred, targ)
            
            # 计算梯度
            grad = torch.autograd.grad(adv_loss, noise, allow_unused=True)[0]
            
            # 如果梯度为None,跳过这次更新
            if grad is None:
                continue
                
            # 更新扰动
            norm = torch.norm(grad)
            if norm != 0 and not torch.isnan(norm):
                noise.data += self.step_size * grad / (norm + self.epsilon)
                
            # 投影到epsilon球内
            noise_norm = torch.norm(noise)
            if noise_norm > self.step_size:
                noise.data = noise.data * self.step_size / noise_norm
                
        # 计算最终的对抗损失
        adv_pred = self.eval_fn(embed + noise.detach())
        adv_loss = self.loss_fn(adv_pred, targ)
        
        return adv_loss
