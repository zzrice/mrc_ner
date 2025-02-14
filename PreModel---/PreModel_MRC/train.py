#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train with valid"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import torch
from transformers import DebertaConfig, DebertaModel
from transformers import AdamW, get_linear_schedule_with_warmup

import random
import argparse
import logging
from tqdm import trange

import utils
from dataloader import NERDataLoader
from model import DeBertaQueryNER
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
parser.add_argument('--sample_ratio', type=float, default=1.0,
                    help="训练数据采样比例，用于快速验证代码。范围[0-1]，默认使用全部数据")

def train(model, data_iterator, optimizer, scheduler, params):
    """训练模型一个epoch"""
    model.train()
    loss_avg = utils.RunningAverage()
    
    # 初始化EMA
    ema = utils.EMA(model, decay=0.999)
    ema.register()
    
    # 初始化FGM
    fgm = utils.FGM(model, emb_name='embeddings', epsilon=1.0)

    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        batch = next(iter(data_iterator))
        batch = tuple(t.to(params.device) for t in batch)
        input_ids, input_mask, segment_ids, start_pos, end_pos, _, _, _ = batch

        # 正常训练
        loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                     start_positions=start_pos, end_positions=end_pos)

        if params.n_gpu > 1 and args.multi_gpu:
            loss = loss.mean()
        if params.gradient_accumulation_steps > 1:
            loss = loss / params.gradient_accumulation_steps

        loss.backward()
        
        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        loss_adv = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                        start_positions=start_pos, end_positions=end_pos)
        if params.n_gpu > 1 and args.multi_gpu:
            loss_adv = loss_adv.mean()
        if params.gradient_accumulation_steps > 1:
            loss_adv = loss_adv / params.gradient_accumulation_steps
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数

        if (step + 1) % params.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_grad)
            optimizer.step()
            scheduler.step()
            # 更新EMA
            ema.update()
            model.zero_grad()

        loss_avg.update(loss.item() * params.gradient_accumulation_steps)
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
    
    return ema

def train_and_evaluate(model, params, restore_file=None):
    """训练模型并每个epoch进行评估."""
    args = parser.parse_args()

    dataloader = NERDataLoader(params)
    train_loader = dataloader.get_dataloader(data_sign='train', sample_ratio=args.sample_ratio)
    val_loader = dataloader.get_dataloader(data_sign='val')

    if args.sample_ratio < 1.0:
        logging.info(f"使用{args.sample_ratio * 100}%的训练数据进行快速验证")

    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("从{}恢复参数".format(restore_path))
        utils.load_checkpoint(restore_path, model)

    model.to(params.device)
    if params.n_gpu > 1 and args.multi_gpu:
        model = torch.nn.DataParallel(model)

    # 准备优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': params.weight_decay_rate},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
         'weight_decay': 0.0}
    ]

    num_training_steps = len(train_loader) * args.epoch_num // params.gradient_accumulation_steps
    num_warmup_steps = int(num_training_steps * params.warmup_prop)

    optimizer = AdamW(optimizer_grouped_parameters, lr=params.learning_rate)
    
    # 使用两个调度器：
    # 1. warmup阶段的线性调度器
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    # 2. 主要训练阶段的动态调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2,
        min_lr=1e-6, verbose=True
    )

    if restore_file is not None:
        restore_path = os.path.join(params.model_dir, args.restore_file + '.pth.tar')
        logging.info("从{}恢复参数".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, args.epoch_num + 1):
        logging.info("Epoch {}/{}".format(epoch, args.epoch_num))
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"当前学习率: {current_lr}")

        # 训练阶段
        ema = train(model, train_loader, optimizer, warmup_scheduler, params)
        
        # 评估阶段
        ema.apply_shadow()
        val_metrics = evaluate(args, model, val_loader, params)
        ema.restore()
        
        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # 根据验证集F1分数调整学习率
        if epoch > num_warmup_steps:  # warmup后才开始使用ReduceLROnPlateau
            scheduler.step(val_f1)

        # 保存模型时使用EMA权重
        if improve_f1 > 0:
            ema.apply_shadow()
            model_to_save = model.module if hasattr(model, 'module') else model
            utils.save_checkpoint({'epoch': epoch + 1,
                                'model': model_to_save,
                                'optim': optimizer,
                                'scheduler': scheduler},
                                is_best=True,
                                checkpoint=params.model_dir)
            ema.restore()
            params.save(params.params_path / 'params.json')

        if improve_f1 > 0:
            logging.info("- 找到新的最佳F1分数: {:.4f}".format(val_f1))
            best_val_f1 = val_f1
            if improve_f1 < params.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
            logging.info("- F1分数未提升: {:.4f}".format(val_f1))

        if (patience_counter > params.patience_num and epoch > params.min_epoch_num) or epoch == args.epoch_num:
            logging.info("最佳验证F1分数: {:05.2f}".format(best_val_f1))
            break

if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.Params(ex_index=args.ex_index)
    utils.set_logger(log_path=os.path.join(params.params_path, 'train.log'), save=True)

    if args.multi_gpu:
        params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
        params.n_gpu = n_gpu
    else:
        torch.cuda.set_device(args.device_id)
        print('current device:', torch.cuda.current_device())
        n_gpu = 1
        params.n_gpu = n_gpu

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logging.info("Model type: 'DeBERTa-MRC'")
    logging.info("device: {}".format(params.device))

    logging.info('Init pre-train model...')
    # 从本地加载配置文件
    config_path = os.path.join(params.bert_model_dir, 'config.json')
    deberta_config = DebertaConfig.from_json_file(config_path)
    # 从本地加载模型
    model = DeBertaQueryNER(config=deberta_config, params=params)
    model.deberta = DebertaModel.from_pretrained(params.bert_model_dir, config=deberta_config)
    logging.info('-done')

    logging.info("Starting training for {} epoch(s)".format(args.epoch_num))
    train_and_evaluate(model, params, args.restore_file)
