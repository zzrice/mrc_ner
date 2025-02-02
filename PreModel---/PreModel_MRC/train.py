#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from NEZHA.model_NEZHA import NEZHAConfig
from NEZHA.NEZHA_utils import torch_init_model
from smart_pytorch import SMARTLoss
import torch.nn as nn

# 参数解析器
import random
import argparse
import logging
from tqdm import trange

import utils
from optimization import BertAdam
from dataloader import NERDataLoader
from model import BertQueryNER
from evaluate import evaluate

# 设定参数
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
parser.add_argument('--ex_index', type=int, default=1, help="实验名称索引")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file containing weights to reload before training")
parser.add_argument('--epoch_num', required=True, type=int,
                    help="指定epoch_num")
parser.add_argument('--multi_gpu', action='store_true', help="是否多GPU")
parser.add_argument('--device_id', type=int, default=0)


def train(model, data_iterator, optimizer, params):
    """训练模型一个epoch
    """
    # 将模型设置为训练模式
    model.train()

    # 记录平均损失
    loss_avg = utils.RunningAverage()

    # 初始化SMART损失
    if params.use_smart:
        # 定义评估函数
        def eval_fn(embeds):
            # 获取原始input_ids对应的embeddings
            original_embeds = model.word_embeddings(input_ids)
            # 计算扰动
            delta = embeds - original_embeds
            # 将扰动添加到原始embeddings
            perturbed_embeds = original_embeds + delta
            # 使用原始input_ids进行前向传播，但在内部替换embeddings
            return model(input_ids=input_ids, token_type_ids=segment_ids,
                       attention_mask=input_mask)
        
        # 定义损失函数
        def loss_fn(pred, targ):
            start_logits, end_logits = pred
            start_positions, end_positions = targ
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            total_loss = (start_loss + end_loss) / 2
            return total_loss

        # 初始化SMART损失函数
        smart_loss_func = SMARTLoss(
            eval_fn=eval_fn,
            loss_fn=loss_fn,
            num_steps=params.smart_num_steps,
            step_size=params.smart_step_size,
            epsilon=params.smart_step_size,
            noise_var=params.smart_noise_var
        )

    # 使用tqdm显示进度条
    # 一个epoch的训练步数等于dataloader的长度
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        # 获取下一个训练批次
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, start_pos, end_pos, _, _, _ = batch

        # 计算标准损失
        loss = model(input_ids=input_ids, token_type_ids=segment_ids, 
                    attention_mask=input_mask,
                    start_positions=start_pos, end_positions=end_pos)

        if params.use_smart:
            # 获取初始embeddings
            embeds = model.word_embeddings(input_ids)
            
            # 计算SMART损失
            smart_loss = smart_loss_func(
                embeds,
                model(input_ids=input_ids, token_type_ids=segment_ids,
                     attention_mask=input_mask),
                (start_pos, end_pos)
            )
            # 合并损失
            loss = loss + params.smart_loss_weight * smart_loss

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()  # 在多GPU上取平均值
        # 梯度累加
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps

        # 反向传播
        loss.backward()

        if (step + 1) % params.gradient_accumulation_steps == 0:
            # 使用计算的梯度进行参数更新
            optimizer.step()
            model.zero_grad()

        # 更新平均损失
        loss_avg.update(loss.item())
        # 设置进度条后缀，显示当前损失值
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))


def train_and_evaluate(model, params, restore_file=None):
    """训练模型并每个epoch进行评估."""
    # 加载参数
    args = parser.parse_args()

    # 加载训练数据和验证数据
    dataloader = NERDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train')
    val_loader = dataloader.get_dataloader(data_sign='val')

    # 如果指定了restore_file，则从restore_file中恢复权重
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("从{}恢复参数".format(restore_path))
        # 读取checkpoint
        model, optimizer = utils.load_checkpoint(restore_path)

    model.to(params.device)
    # 如果使用多GPU，则将模型并行化
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # 准备优化器
    # 微调
    # 获取模型参数
    param_optimizer = list(model.named_parameters())
    # 预训练模型参数
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # 下游任务模型参数
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    # 不进行衰减的参数
    no_decay = ['bias', 'LayerNorm', 'layer_norm', 'dym_weight']
    # 将参数分组
    optimizer_grouped_parameters = [
        # 预训练模型参数
        # 衰减
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.fin_tuning_lr
         },
        # 不衰减
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.fin_tuning_lr
         },
        # 下游任务模型
        # 衰减
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate, 'lr': params.downstream_lr
         },
        # 不衰减
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': params.downstream_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // params.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=params.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=params.clip_grad)

    # 如果指定了restore_file，则从restore_file中恢复权重
    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("从{}恢复参数".format(restore_path))
        # 读取checkpoint
        utils.load_checkpoint(restore_path, model, optimizer)

    # 设置最佳验证F1分数和耐心计数器
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epoch_num + 1):
        # 运行一个epoch
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # 在训练集上训练一个epoch
        train(model, train_loader, optimizer, params)

        # 在训练集和验证集上评估一个epoch
        # train_metrics = evaluate(model, train_loader, params, mark='Train',
        #                          verbose=True)  # Dict['loss', 'f1']
        val_metrics = evaluate(args, model, val_loader, params)  # Dict['loss', 'f1']
        # 验证集F1分数
        val_f1 = val_metrics['f1']
        # F1分数提升值
        improve_f1 = val_f1 - best_val_f1

        # 保存网络权重
        model_to_save = model.module if hasattr(model, 'module') else model  # 仅保存模型本身
        optimizer_to_save = optimizer
        utils.save_checkpoint({'epoch': epoch + 1,
                               'model': model_to_save,
                               'optim': optimizer_to_save},
                              is_best=improve_f1 > 0,
                              checkpoint=params.model_dir)
        params.save(params.params_path / 'params.json')

        # 基于patience停止训练
        if improve_f1 > 0:
            logging.info("- 找到新的最佳F1分数")
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1

        # 早停和记录最佳F1分数
        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("最佳验证F1分数: {:05.2f}".format(best_val_f1))
            break


if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化参数对象，根据实验编号创建参数配置
    params = utils.Params(ex_index=args.ex_index)
    
    # 设置日志记录器，将日志保存到指定路径
    utils.set_logger(log_path=os.path.join(params.params_path, 'train.log'), save=True)

    # 设备配置
    if args.multi_gpu:
        # 多GPU模式：自动检测可用GPU数量
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        # 单GPU模式：手动指定GPU设备
        torch.cuda.set_device(args.device_id)
        # 打印当前使用的GPU设备
        print('current device:', torch.cuda.current_device())
        n_gpu = 1
        params.n_gpu = n_gpu

    # 设置随机种子以保证实验可重复性
    random.seed(args.seed)  # Python随机数种子
    torch.manual_seed(args.seed)  # PyTorch CPU随机种子
    params.seed = args.seed  # 将种子保存到参数对象
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)  # 设置所有GPU的随机种子

    # 初始化日志记录
    logging.info("Model type: 'NEZHA-MRC'")  # 记录模型类型
    logging.info("device: {}".format(params.device))  # 记录使用的设备

    # 准备模型
    logging.info('Init pre-train model...')

    # 从配置文件加载NEZHA模型配置
    bert_config = NEZHAConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))

    # 初始化BERTQueryNER模型
    model = BertQueryNER(config=bert_config, params=params)
    # 加载预训练权重

    torch_init_model(model, os.path.join(params.bert_model_dir, 'pytorch_model.bin'))
    logging.info('-done')  # 模型初始化完成

    # 开始训练和评估
    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    # 调用训练和评估主函数
    train_and_evaluate(model, params, args.restore_file)
