#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dataloader.py utils"""
import re
import json
import logging
import random
import torch


def split_text(text, max_len, split_pat=r'([，。]”?)', greedy=False):
    """文本分片
    将超过长度的文本分片成多段满足最大长度要求的最长连续子文本
    约束条件：1）每个子文本最大长度不超过max_len；
             2）所有的子文本的合集要能覆盖原始文本。
    Arguments:
        text {str} -- 原始文本
        max_len {int} -- 最大长度

    Keyword Arguments:
        split_pat {str or re pattern} -- 分割符模式 (default: {SPLIT_PAT})
        greedy {bool} -- 是否选择贪婪模式 (default: {False})
                         贪婪模式：在满足约束条件下，选择子文本最多的分割方式
                         非贪婪模式：在满足约束条件下，选择冗余度最小且交叉最为均匀的分割方式

    Returns:
        tuple -- 返回子文本列表以及每个子文本在原始文本中对应的起始位置列表

    Examples:
        text = '今夕何夕兮，搴舟中流。今日何日兮，得与王子同舟。蒙羞被好兮，不訾诟耻。心几烦而不绝兮，得知王子。山有木兮木有枝，心悦君兮君不知。'
        sub_texts, starts = split_text(text, maxlen=30, greedy=False)
        for sub_text in sub_texts:
            print(sub_text)
        print(starts)
        for start, sub_text in zip(starts, sub_texts):
            if text[start: start + len(sub_text)] != sub_text:
            print('Start indice is wrong!')
            break
    """
    # 文本小于max_len则不分割
    if len(text) <= max_len:
        return [text], [0]
    # 分割字符串
    segs = re.split(split_pat, text)
    # init
    sentences = []
    # 将分割后的段落和分隔符组合
    for i in range(0, len(segs) - 1, 2):
        sentences.append(segs[i] + segs[i + 1])
    if segs[-1]:
        sentences.append(segs[-1])
    n_sentences = len(sentences)
    sent_lens = [len(s) for s in sentences]

    # 所有满足约束条件的最长子片段
    alls = []
    for i in range(n_sentences):
        length = 0
        sub = []
        for j in range(i, n_sentences):
            if length + sent_lens[j] <= max_len or not sub:
                sub.append(j)
                length += sent_lens[j]
            else:
                break
        alls.append(sub)
        # 将最后一个段落加入
        if j == n_sentences - 1:
            if sub[-1] != j:
                alls.append(sub[1:] + [j])
            break

    if len(alls) == 1:
        return [text], [0]

    if greedy:
        # 贪婪模式返回所有子文本
        sub_texts = [''.join([sentences[i] for i in sub]) for sub in alls]
        starts = [0] + [sum(sent_lens[:i]) for i in range(1, len(alls))]
        return sub_texts, starts
    else:
        # 用动态规划求解满足要求的最优子片段集
        DG = {}  # 有向图
        N = len(alls)
        for k in range(N):
            tmplist = list(range(k + 1, min(alls[k][-1] + 1, N)))
            if not tmplist:
                tmplist.append(k + 1)
            DG[k] = tmplist

        routes = {}
        routes[N] = (0, -1)
        for i in range(N - 1, -1, -1):
            templist = []
            for j in DG[i]:
                cross = set(alls[i]) & (set(alls[j]) if j < len(alls) else set())
                w_ij = sum([sent_lens[k] for k in cross]) ** 2  # 第i个节点与第j个节点交叉度
                w_j = routes[j][0]  # 第j个子问题的值
                w_i_ = w_ij + w_j
                templist.append((w_i_, j))
            routes[i] = min(templist)

        sub_texts, starts = [''.join([sentences[i] for i in alls[0]])], [0]
        k = 0
        while True:
            k = routes[k][1]
            sub_texts.append(''.join([sentences[i] for i in alls[k]]))
            starts.append(sum(sent_lens[: alls[k][0]]))
            if k == N - 1:
                break

    return sub_texts, starts


def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text.
    """
    text = text.strip()
    # 内容为空则返回空列表
    if not text:
        return []
    tokens = list(text)
    return tokens


class InputExample(object):
    """a single set of samples of data_src
    """

    def __init__(self,
                 query_item,
                 context_item,
                 start_position=None,
                 end_position=None,
                 en_cate=None):
        self.query_item = query_item
        self.context_item = context_item
        self.start_position = start_position
        self.end_position = end_position
        self.en_cate = en_cate


class InputFeatures(object):
    """
    Desc:
        a single set of features of Examples
    Args:
        start_position: start position is a list of symbol
        end_position: end position is a list of symbol
    """

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 en_cate,
                 split_to_original_id,
                 example_id,
                 start_position=None,
                 end_position=None,
                 ):
        self.input_mask = input_mask
        self.input_ids = input_ids
        self.en_cate = en_cate
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position

        # use to split
        self.split_to_original_id = split_to_original_id
        self.example_id = example_id


def read_examples(input_file):
    """read data_src to InputExamples
    :return examples (List[InputExample])
    """
    # read json file
    with open(input_file, "r", encoding='utf-8') as f:
        input_data = json.load(f)
    # get InputExample class
    examples = []
    for entry in input_data:
        query_item = entry["query"]
        context_item = entry["context"]
        start_position = entry["start_position"]
        end_position = entry["end_position"]
        en_cate = entry["entity_type"]

        example = InputExample(query_item=query_item,
                               context_item=context_item,
                               start_position=start_position,
                               end_position=end_position,
                               en_cate=en_cate)
        examples.append(example)
    print("InputExamples:", len(examples))
    return examples


def augment_data(text, query, start_pos, end_pos, max_length, augment_ratio=0.5, window_stride=0.8):
    """数据增强方法
    
    Args:
        text: 原始文本
        query: 查询文本
        start_pos: 起始位置
        end_pos: 结束位置
        max_length: 最大序列长度
        augment_ratio: 数据增强比例，控制增强样本数量
        window_stride: 滑动窗口的步长比例
    
    Returns:
        list: 增强后的数据列表，每个元素为(text, query, start_pos, end_pos)
    """
    augmented_data = []
    
    # 1. 滑动窗口策略
    window_size = max_length - len(query) - 3  # 预留[CLS], [SEP], [SEP]的位置
    stride = int(window_size * window_stride)  # 使用window_stride控制步长
    
    for i in range(0, len(text), stride):
        window_text = text[i:i + window_size]
        if len(window_text) < window_size // 2:  # 窗口太小则跳过
            continue
        
        # 调整答案位置
        new_start = start_pos - i
        new_end = end_pos - i
        
        # 检查答案是否在当前窗口内
        if 0 <= new_start < len(window_text) and 0 <= new_end < len(window_text):
            # 使用augment_ratio控制是否添加这个增强样本
            if random.random() < augment_ratio:
                augmented_data.append((window_text, query, new_start, new_end))
    
    # 2. 动态最大长度策略
    if len(text) > max_length:
        # 确保答案在文本中间
        answer_center = (start_pos + end_pos) // 2
        half_length = (max_length - len(query) - 3) // 2
        
        text_start = max(0, answer_center - half_length)
        text_end = min(len(text), answer_center + half_length)
        
        # 如果答案在文本开头或结尾附近，调整窗口位置
        if text_start == 0:
            text_end = min(len(text), max_length - len(query) - 3)
        elif text_end == len(text):
            text_start = max(0, len(text) - (max_length - len(query) - 3))
        
        window_text = text[text_start:text_end]
        new_start = start_pos - text_start
        new_end = end_pos - text_start
        
        if 0 <= new_start < len(window_text) and 0 <= new_end < len(window_text):
            # 使用augment_ratio控制是否添加这个增强样本
            if random.random() < augment_ratio:
                augmented_data.append((window_text, query, new_start, new_end))
    
    return augmented_data


def convert_examples_to_features(params, examples, tokenizer, greed_split=False, data_sign='train'):
    """将InputExamples转换为InputFeatures
    """
    features = []
    total_original_samples = len(examples)
    total_augmented_samples = 0
    augmented_stats = {
        'window_augmented': 0,
        'dynamic_length_augmented': 0
    }
    
    logging.info(f"开始处理{data_sign}数据集，原始样本数量: {total_original_samples}")
    
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.query_item)
        
        # 对于训练数据，使用数据增强
        if data_sign == 'train':
            augmented_samples = augment_data(
                example.context_item,
                example.query_item,
                example.start_position,
                example.end_position,
                params.max_seq_length,
                params.augment_ratio,
                params.window_stride
            )
            
            # 记录增强样本数量
            if len(augmented_samples) > 0:
                total_augmented_samples += len(augmented_samples)
                
                # 记录每种增强方法的统计信息
                for aug_text, _, _, _ in augmented_samples:
                    if len(aug_text) < len(example.context_item):
                        augmented_stats['window_augmented'] += 1
                    else:
                        augmented_stats['dynamic_length_augmented'] += 1
            
            # 每处理100个样本输出一次日志
            if (example_index + 1) % 100 == 0:
                logging.info(f"已处理 {example_index + 1}/{total_original_samples} 个原始样本")
                logging.info(f"当前增强样本数量: {total_augmented_samples}")
                logging.info(f"滑动窗口增强: {augmented_stats['window_augmented']}")
                logging.info(f"动态长度增强: {augmented_stats['dynamic_length_augmented']}")
                logging.info(f"当前内存使用: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            
            # 合并原始样本和增强样本
            all_samples = [(example.context_item, example.query_item, example.start_position, example.end_position)] + augmented_samples
        else:
            # 验证和测试数据不做增强
            all_samples = [(example.context_item, example.query_item, example.start_position, example.end_position)]
            
        # 处理所有样本（原始+增强）
        for text, query, start_position, end_position in all_samples:
            # 转换为特征
            feature = convert_single_example_to_feature(
                params,
                tokenizer.tokenize(query),
                tokenizer.tokenize(text),
                start_position,
                end_position,
                tokenizer,
                example_index,
                0  # 由于已经在augment_data中处理了切分，这里使用0
            )
            if feature is not None:
                features.append(feature)
    
    # 输出最终统计信息
    if data_sign == 'train':
        logging.info("="*50)
        logging.info("数据增强统计信息：")
        logging.info(f"原始样本数量: {total_original_samples}")
        logging.info(f"增强后总样本数量: {len(features)}")
        logging.info(f"新增样本数量: {total_augmented_samples}")
        logging.info(f"滑动窗口增强样本数量: {augmented_stats['window_augmented']}")
        logging.info(f"动态长度增强样本数量: {augmented_stats['dynamic_length_augmented']}")
        logging.info(f"平均每个样本增强数量: {total_augmented_samples/total_original_samples:.2f}")
        logging.info("="*50)
                    
    return features

def convert_single_example_to_feature(params, query_tokens, context_tokens, start_position, end_position, tokenizer, example_index, split_index):
    """转换单个样本为特征
    """
    tokens = []
    segment_ids = []
    
    # 添加[CLS]
    tokens.append("[CLS]")
    segment_ids.append(0)
    
    # 添加query tokens
    for token in query_tokens:
        tokens.append(token)
        segment_ids.append(0)
    
    # 添加[SEP]
    tokens.append("[SEP]")
    segment_ids.append(0)
    
    # 添加context tokens
    for token in context_tokens:
        tokens.append(token)
        segment_ids.append(1)
    
    # 添加最后的[SEP]
    tokens.append("[SEP]")
    segment_ids.append(1)
    
    # 转换为ids
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    
    # 补零
    while len(input_ids) < params.max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    
    # 调整答案位置
    start_position += len(query_tokens) + 2  # +2是因为[CLS]和第一个[SEP]
    end_position += len(query_tokens) + 2
    
    # 创建特征对象
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        start_position=start_position,
        end_position=end_position,
        en_cate=tag2idx[example.en_cate],
        split_to_original_id=split_index,
        example_id=example_index
    )
    
    return feature
