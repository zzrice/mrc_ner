# /usr/bin/env python
# coding=utf-8
"""Dataloader"""

import os
import logging

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from NEZHA.tokenization import BertTokenizer

from dataloader_utils import read_examples, convert_examples_to_features


class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]


class NERDataLoader(object):
    """dataloader
    """

    def __init__(self, params):
        self.params = params

        self.train_batch_size = params.train_batch_size
        self.val_batch_size = params.val_batch_size
        self.test_batch_size = params.test_batch_size

        self.data_dir = params.data_dir
        self.max_seq_length = params.max_seq_length
        self.tokenizer = BertTokenizer(vocab_file=os.path.join(params.bert_model_dir, 'vocab.txt'),
                                       do_lower_case=True)
        # 保存数据(Bool)
        self.data_cache = params.data_cache

    @staticmethod
    def collate_fn(features):
        """将InputFeatures转换为Tensor
        Args:
            features (List[InputFeatures])
        Returns:
            tensors (List[Tensors])
        """
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        gold_start = torch.tensor([f.start_position for f in features], dtype=torch.long)
        gold_end = torch.tensor([f.end_position for f in features], dtype=torch.long)
        en_cate = torch.tensor([f.en_cate for f in features], dtype=torch.long)

        # use to split text
        split_to_ori = torch.tensor([f.split_to_original_id for f in features], dtype=torch.long)
        example_ids = torch.tensor([f.example_id for f in features], dtype=torch.long)
        tensors = [input_ids, input_mask, segment_ids, gold_start, gold_end, en_cate, split_to_ori, example_ids]
        return tensors

    def get_features(self, data_sign):
        """convert InputExamples to InputFeatures
        :param data_sign: 'train', 'val' or 'test'
        :return: features (List[InputFeatures]):
        """
        print("="*20 + f"加载{data_sign}数据" + "="*20)
        # get examples
        if data_sign in ("train", "val", "test", "pseudo"):
            examples = read_examples(os.path.join(self.data_dir, f'{data_sign}.data'))
        else:
            raise ValueError("请注意数据集只能是 train/val/test！")
        
        # get features
        cache_path = os.path.join(self.data_dir, "{}.cache.{}".format(data_sign, str(self.max_seq_length)))
        
        if os.path.exists(cache_path) and self.data_cache:
            logging.info(f"从缓存加载{data_sign}数据: {cache_path}")
            features = torch.load(cache_path)
        else:
            logging.info(f"开始处理{data_sign}数据...")
            logging.info(f"最大序列长度: {self.max_seq_length}")
            if data_sign == 'train':
                logging.info("将对训练数据进行增强处理...")
                logging.info("增强方法: 1.滑动窗口 2.动态长度调整")
                
            # 生成特征
            features = convert_examples_to_features(
                self.params, 
                examples, 
                self.tokenizer, 
                greed_split=False,
                data_sign=data_sign
            )
            
            # 保存数据
            if self.data_cache:
                logging.info(f"保存处理后的数据到缓存: {cache_path}")
                torch.save(features, cache_path)
            
        logging.info(f"最终得到{len(features)}个特征")
        print("="*50)
        return features

    def get_dataloader(self, data_sign="train", sample_ratio=1.0):
        """构造数据加载器
        
        Args:
            data_sign: str, 'train', 'val' or 'test'
            sample_ratio: float, 训练数据采样比例，范围[0-1]，仅在data_sign为'train'时有效
            
        Returns:
            dataloader: DataLoader对象
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign)
        
        # 对训练数据进行采样
        if data_sign == "train" and sample_ratio < 1.0:
            num_samples = int(len(features) * sample_ratio)
            # 使用相同的随机种子以保证可重复性
            torch.manual_seed(self.params.seed)
            indices = torch.randperm(len(features))[:num_samples]
            features = [features[idx] for idx in indices]
            logging.info(f"采样后的训练数据数量: {len(features)}")
        
        dataset = FeatureDataset(features)
        print(f"{len(features)} {data_sign} data loaded!")
        print("=*=" * 10)

        # construct dataloader
        # RandomSampler(dataset) or SequentialSampler(dataset)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn)
        elif data_sign in ("test", "pseudo"):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn)
        else:
            raise ValueError("please notice that the data can only be train/val/test !!")
        return dataloader

    def augment_data(self, text, query, start_pos, end_pos, max_length):
        """数据增强方法
        
        Args:
            text: 原始文本
            query: 查询文本
            start_pos: 起始位置
            end_pos: 结束位置
            max_length: 最大序列长度
        
        Returns:
            list: 增强后的数据列表，每个元素为(text, query, start_pos, end_pos)
        """
        augmented_data = []
        
        # 1. 滑动窗口策略
        window_size = max_length - len(query) - 3  # 预留[CLS], [SEP], [SEP]的位置
        stride = window_size // 2
        
        for i in range(0, len(text), stride):
            window_text = text[i:i + window_size]
            if len(window_text) < window_size // 2:  # 窗口太小则跳过
                continue
            
            # 调整答案位置
            new_start = start_pos - i
            new_end = end_pos - i
            
            # 检查答案是否在当前窗口内
            if 0 <= new_start < len(window_text) and 0 <= new_end < len(window_text):
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
                augmented_data.append((window_text, query, new_start, new_end))
        
        return augmented_data

    def convert_examples_to_features(self, examples, data_sign='train'):
        """将样本转换为特征
        """
        features = []
        
        for (example_id, example) in enumerate(examples):
            query = example.query
            context = example.context
            start_pos = example.start_position
            end_pos = example.end_position
            
            if data_sign == 'train':
                # 对训练数据进行增强
                augmented_samples = self.augment_data(
                    context, query, start_pos, end_pos, 
                    self.max_seq_length
                )
                
                # 处理原始样本和增强样本
                all_samples = [(context, query, start_pos, end_pos)] + augmented_samples
            else:
                # 验证和测试数据不做增强
                all_samples = [(context, query, start_pos, end_pos)]
            
            for text, q, s_pos, e_pos in all_samples:
                # 原有的特征转换逻辑
                query_tokens = self.tokenizer.tokenize(q)
                context_tokens = self.tokenizer.tokenize(text)
                
                # ... 其余特征转换代码保持不变 ...
                
        return features


if __name__ == '__main__':
    from utils import Params

    params = Params()
    datalodaer = NERDataLoader(params)
    feats = datalodaer.get_features(data_sign='train')
    print(len(feats))
    print(feats[0].input_ids)
    print(feats[1].input_ids)
    print(feats[0].split_to_original_id)
    print(feats[1].split_to_original_id)
