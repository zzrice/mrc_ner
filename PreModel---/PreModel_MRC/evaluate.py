#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate"""
import logging
from tqdm import tqdm

import torch

import utils
from utils import EN2QUERY
from metrics import classification_report, f1_score, accuracy_score


def pointer2bio(start_labels, end_labels, en_cate):
    """convert (begin, end, span) label to bio label. for single sample.
    :return: bio_labels List[str]: 实体序列（单样本）
    """
    # init
    bio_labels = len(start_labels) * ["O"]

    # 取出start idx和end idx
    start_labels = [idx for idx, tmp in enumerate(start_labels) if tmp != 0]
    end_labels = [idx for idx, tmp in enumerate(end_labels) if tmp != 0]

    # 打start标
    for start_item in start_labels:
        bio_labels[start_item] = "B-{}".format(en_cate)

    # 打I标
    for tmp_start in start_labels:
        # 取出在start position后的end position
        tmp_end = [tmp for tmp in end_labels if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            # 取出距离最近的end
            tmp_end = min(tmp_end)
        # 如果匹配则标记为实体
        if tmp_start != tmp_end:
            for i in range(tmp_start + 1, tmp_end + 1):
                bio_labels[i] = "I-{}".format(en_cate)
        # 单字实体
        else:
            bio_labels[tmp_end] = "B-{}".format(en_cate)

    return bio_labels


def evaluate(args, model, eval_dataloader, params):
    """模型评估函数
    
    该函数用于在验证集或测试集上评估模型性能，计算损失、F1分数和准确率等指标
    
    Args:
        args: 命令行参数
        model: 待评估的模型
        eval_dataloader: 评估数据加载器
        params: 模型参数配置
        
    Returns:
        metrics: 包含评估指标的字典
    """
    # 将模型设置为评估模式
    model.eval()
    
    # 初始化运行平均损失计算器
    loss_avg = utils.RunningAverage()
    
    # 初始化预测结果和真实结果列表
    pre_result = []  # 存储模型预测的BIO标签
    gold_result = []  # 存储真实的BIO标签

    # 遍历评估数据集
    for batch in tqdm(eval_dataloader, unit='Batch', ascii=True):
        # 将数据移动到指定设备（如GPU）
        batch = tuple(t.to(params.device) for t in batch)
        # 解包batch数据
        input_ids, input_mask, segment_ids, start_pos, end_pos, en_cate, _, _ = batch

        # 禁用梯度计算
        with torch.no_grad():
            # 计算模型损失
            loss = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                         start_positions=start_pos, end_positions=end_pos)
            # 如果使用多GPU，对损失取平均
            if params.n_gpu > 1 and args.multi_gpu:
                loss = loss.mean()
            # 更新运行平均损失
            loss_avg.update(loss.item())

            # 模型推理，获取预测的start和end位置
            start_pre, end_pre = model(input_ids=input_ids,
                                       token_type_ids=segment_ids, attention_mask=input_mask)

        # 将真实标签转移到CPU并转换为列表
        start_pos = start_pos.to("cpu").numpy().tolist()
        end_pos = end_pos.to("cpu").numpy().tolist()
        input_mask = input_mask.to('cpu').numpy().tolist()
        en_cate = en_cate.to("cpu").numpy().tolist()

        # 将预测结果转移到CPU并转换为列表
        start_pre = start_pre.detach().cpu().numpy().tolist()
        end_pre = end_pre.detach().cpu().numpy().tolist()

        # 创建类别索引到标签的映射
        cate_idx2label = {idx: value for idx, value in enumerate(params.tag_list)}

        # 处理每个样本的预测和真实标签
        for start_p, end_p, start_g, end_g, input_mask_s, en_cate_s in zip(start_pre, end_pre,
                                                                           start_pos, end_pos,
                                                                           input_mask, en_cate):
            # 获取当前样本的类别字符串
            en_cate_str = cate_idx2label[en_cate_s]
            # 计算问题长度
            q_len = len(EN2QUERY[en_cate_str])
            # 计算有效文本长度
            act_len = sum(input_mask_s[q_len + 2:-1])
            # 将预测的start和end位置转换为BIO标签
            pre_bio_labels = pointer2bio(start_p[q_len + 2:q_len + 2 + act_len],
                                         end_p[q_len + 2:q_len + 2 + act_len],
                                         en_cate=en_cate_str)
            # 将真实的start和end位置转换为BIO标签
            gold_bio_labels = pointer2bio(start_g[q_len + 2:q_len + 2 + act_len],
                                          end_g[q_len + 2:q_len + 2 + act_len],
                                          en_cate=en_cate_str)
            # 保存结果
            pre_result.append(pre_bio_labels)
            gold_result.append(gold_bio_labels)

    # 计算评估指标
    f1 = f1_score(y_true=gold_result, y_pred=pre_result)  # 计算F1分数
    acc = accuracy_score(y_true=gold_result, y_pred=pre_result)  # 计算准确率

    # 组织评估结果
    metrics = {'loss': loss_avg(), 'f1': f1, 'acc': acc}
    # 格式化评估指标字符串
    metrics_str = "; ".join("{}: {:05.2f}".format(k, v) for k, v in metrics.items())
    # 记录评估结果
    logging.info("- {} metrics: ".format('Val') + metrics_str)
    # 生成分类报告
    report = classification_report(y_true=gold_result, y_pred=pre_result)
    logging.info(report)

    return metrics
