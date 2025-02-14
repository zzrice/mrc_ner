# /usr/bin/env python
# coding=utf-8
"""Thanks to https://github.com/chakki-works/seqeval
Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
"""
# 引入新版本特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np


def get_entities(seq, suffix=False):
    """从标签序列中提取实体信息

    该方法用于从BIO/BIOES等序列标注格式中提取实体信息，返回实体的类型及其在序列中的起止位置。

    Args:
        seq (list): 标签序列，可以是嵌套列表
        suffix (bool): 是否使用后缀模式。默认为False，表示使用前缀模式（如B-、I-等）

    Returns:
        list: 包含实体信息的列表，每个元素为三元组 (实体类型, 起始位置, 结束位置)

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # 处理嵌套列表的情况，将其展平为一维列表
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    # 初始化变量
    prev_tag = 'O'  # 前一个标签
    prev_type = ''  # 前一个实体类型
    begin_offset = 0  # 实体起始位置
    chunks = []  # 存储提取的实体信息

    # 遍历序列，注意在末尾添加'O'作为哨兵值
    for i, chunk in enumerate(seq + ['O']):
        # 根据suffix参数决定如何解析标签
        if suffix:
            tag = chunk[-1]  # 使用后缀模式，取最后一个字符作为标签
            type_ = chunk.split('-')[0]  # 实体类型在'-'之前
        else:
            tag = chunk[0]  # 使用前缀模式，取第一个字符作为标签
            type_ = chunk.split('-')[-1]  # 实体类型在'-'之后

        # 判断是否到达实体结尾
        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))  # 保存当前实体信息
        
        # 判断是否开始新实体
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i  # 更新实体起始位置

        # 更新前一个标签和实体类型
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average='micro', digits=2, suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
    r = 100 * nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def classification_report(y_true, y_pred, digits=2, suffix=False):
    """生成分类评估报告，展示主要分类指标
    
    该函数用于生成一个文本报告，展示每个类别的精确率(precision)、召回率(recall)和F1分数，
    以及整体的加权平均值。

    Args:
        y_true : 2d array. 真实标签值，正确的目标值
        y_pred : 2d array. 预测标签值，由分类器返回的预测结果
        digits : int. 输出浮点数的精度位数，默认为2
        suffix : bool. 是否使用后缀模式，默认为False

    Returns:
        report : string. 包含每个类别的精确率、召回率、F1分数和支持数的文本摘要

    Examples:
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
        avg / total       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    # 获取真实和预测的实体集合
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    # 初始化变量
    name_width = 0  # 用于存储最长的类别名称长度
    d1 = defaultdict(set)  # 存储真实实体的字典
    d2 = defaultdict(set)  # 存储预测实体的字典

    # 处理真实实体
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))  # 按类别分组存储实体
        name_width = max(name_width, len(e[0]))  # 更新最大类别名称长度

    # 处理预测实体
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))  # 按类别分组存储实体

    # 设置报告格式
    last_line_heading = 'avg / total'  # 最后一行标题
    width = max(name_width, len(last_line_heading), digits)  # 计算列宽

    # 设置表头
    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    # 设置行格式
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    # 初始化统计列表
    ps, rs, f1s, s = [], [], [], []

    # 计算每个类别的指标
    for type_name, true_entities in d1.items():
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)  # 正确预测的实体数
        nb_pred = len(pred_entities)  # 预测的实体数
        nb_true = len(true_entities)  # 真实的实体数

        # 计算精确率、召回率和F1分数
        p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 100 * nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        # 将结果添加到报告中
        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        # 保存统计结果
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += u'\n'

    # 计算加权平均值
    report += row_fmt.format(last_line_heading,
                             np.average(ps, weights=s),  # 加权平均精确率
                             np.average(rs, weights=s),  # 加权平均召回率
                             np.average(f1s, weights=s),  # 加权平均F1分数
                             np.sum(s),  # 总支持数
                             width=width, digits=digits)

    return report


def has_overlap(span1, span2):
    """检查两个区间是否满足松弛匹配条件
    
    Args:
        span1: tuple. (start, end)
        span2: tuple. (start, end)
        
    Returns:
        bool. 是否满足松弛匹配条件
    """
    # 根据定义：max(s_i.pos_b, g_j.pos_b) ≤ min(s_i.pos_e, g_j.pos_e)
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])


def relaxed_f1_score(y_true, y_pred, average='micro', digits=2, suffix=False):
    """计算松弛匹配的F1分数
    
    松弛匹配的定义：
    1. s_i.d = g_j.d (类型相同)
    2. max(s_i.pos_b, g_j.pos_b) ≤ min(s_i.pos_e, g_j.pos_e) (位置有重叠)
    3. s_i.c = g_j.c (内容相同)
    
    Args:
        y_true: 2d array. 真实标签序列
        y_pred: 2d array. 预测标签序列
        average: str. 平均方式，默认为'micro'
        digits: int. 结果保留的小数位数
        suffix: bool. 是否使用后缀模式
        
    Returns:
        score: float. 松弛F1分数
    """
    true_entities = get_entities(y_true, suffix)
    pred_entities = get_entities(y_pred, suffix)
    
    # 按类型分组实体
    true_entities_by_type = defaultdict(list)
    pred_entities_by_type = defaultdict(list)
    
    for entity in true_entities:
        true_entities_by_type[entity[0]].append((entity[1], entity[2]))
    
    for entity in pred_entities:
        pred_entities_by_type[entity[0]].append((entity[1], entity[2]))
    
    nb_correct = 0
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)
    
    # 对每种实体类型进行匹配
    for entity_type in true_entities_by_type:
        true_entities_spans = true_entities_by_type[entity_type]
        pred_entities_spans = pred_entities_by_type[entity_type]
        
        # 对于每个预测的实体，检查是否有满足松弛匹配条件的真实实体
        for pred_span in pred_entities_spans:
            for true_span in true_entities_spans:
                # 检查是否满足松弛匹配条件
                if has_overlap(pred_span, true_span):
                    nb_correct += 1
                    break
    
    p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
    r = 100 * nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0
    
    return score


def relaxed_classification_report(y_true, y_pred, digits=2, suffix=False):
    """生成宽松匹配的分类评估报告
    
    Args:
        y_true: 2d array. 真实标签序列
        y_pred: 2d array. 预测标签序列
        digits: int. 结果保留的小数位数
        suffix: bool. 是否使用后缀模式
        
    Returns:
        report: str. 评估报告
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)

    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))

    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    last_line_heading = 'avg / total (relaxed)'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format('', *headers, width=width) + '\n\n'
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    for type_name in sorted(d1.keys()):
        true_entities_spans = d1[type_name]
        pred_entities_spans = d2[type_name]
        
        nb_correct = 0
        nb_pred = len(pred_entities_spans)
        nb_true = len(true_entities_spans)

        for pred_span in pred_entities_spans:
            for true_span in true_entities_spans:
                if has_overlap(pred_span, true_span):
                    nb_correct += 1
                    break
        
        p = 100 * nb_correct / nb_pred if nb_pred > 0 else 0
        r = 100 * nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    report += '\n'

    # 计算平均值
    if len(s) == 0:
        avg_p = 0
        avg_r = 0
        avg_f1 = 0
        total_support = 0
    else:
        avg_p = np.average(ps, weights=s)
        avg_r = np.average(rs, weights=s)
        avg_f1 = np.average(f1s, weights=s)
        total_support = np.sum(s)

    report += row_fmt.format(last_line_heading,
                           avg_p, avg_r, avg_f1, total_support,
                           width=width, digits=digits)

    return report
