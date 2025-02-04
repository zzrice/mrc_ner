#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
"""Downstream task model for DeBERTa MRC NER"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DebertaModel, DebertaPreTrainedModel
from utils import initial_parameter

class MultiLossLayer(nn.Module):
    """implementation of "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics"
    """
    def __init__(self, num_loss):
        """
        Args:
            num_loss (int): number of multi-task loss
        """
        super(MultiLossLayer, self).__init__()
        # 使用更保守的初始化范围，并添加值域限制
        self.log_sigmas = nn.Parameter(torch.zeros(num_loss).clamp(-1, 1), requires_grad=True)

    def get_loss(self, loss_set):
        """
        Args:
            loss_set (Tensor): multi-task loss (num_loss,)
        """
        # 添加值域限制
        clamped_log_sigmas = torch.clamp(self.log_sigmas, -1, 1)
        # 通过exp确保sigma为正值，并且增长受控
        sigmas_sq = torch.exp(2 * clamped_log_sigmas)
        
        # 确保 loss_set 非负
        loss_set = torch.abs(loss_set)
        
        # 添加正则化项
        reg_term = 0.1 * torch.mean(sigmas_sq)
        
        # loss part with regularization
        loss = torch.sum(loss_set / (2 * sigmas_sq) + clamped_log_sigmas) + reg_term
        return loss


class DeBertaQueryNER(DebertaPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.deberta = DebertaModel(config)
        
        # start and end position layer
        self.start_outputs = nn.Linear(config.hidden_size, 2)
        self.end_outputs = nn.Linear(config.hidden_size, 2)

        # 动态权重
        self.fusion_layers = params.fusion_layers
        self.dym_weight = nn.Parameter(torch.ones((self.fusion_layers, 1, 1, 1)),
                                       requires_grad=True)

        # self-adaption weight loss
        self.multi_loss_layer = MultiLossLayer(num_loss=2)

        # init weights
        self.init_weights()
        self.init_param()

    def init_param(self):
        initial_parameter(self.start_outputs)
        initial_parameter(self.end_outputs)
        nn.init.xavier_normal_(self.dym_weight)

    def get_dym_layer(self, outputs):
        """获取动态权重融合后的deberta output"""
        hidden_stack = torch.stack(outputs.hidden_states[-self.fusion_layers:],
                                   dim=0)  # (deberta_layer, batch_size, sequence_length, hidden_size)
        sequence_output = torch.sum(hidden_stack * self.dym_weight,
                                    dim=0)  # (batch_size, seq_len, hidden_size[embedding_dim])
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                start_positions=None, end_positions=None):
        """
        Args:
            start_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
            end_positions: (batch x max_len x 1)
                [[0, 1, 0, 0, 1, 0, 1, 0, 0, ], [0, 1, 0, 0, 1, 0, 1, 0, 0, ]] 
        """
        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True
        )

        sequence_output = self.get_dym_layer(outputs)
        batch_size, seq_len, hid_size = sequence_output.size()

        start_logits = self.start_outputs(sequence_output)
        end_logits = self.end_outputs(sequence_output)

        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            start_loss = loss_fct(start_logits.view(-1, 2), start_positions.view(-1))
            end_loss = loss_fct(end_logits.view(-1, 2), end_positions.view(-1))
            
            start_loss = torch.sum(start_loss * token_type_ids.view(-1)) / batch_size
            end_loss = torch.sum(end_loss * token_type_ids.view(-1)) / batch_size
            
            # 确保基础损失非负
            start_loss = torch.abs(start_loss)
            end_loss = torch.abs(end_loss)
            
            total_loss = self.multi_loss_layer.get_loss(torch.cat([start_loss.view(1), end_loss.view(1)]))
            
            # 添加损失值检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"警告：损失值异常 - start_loss: {start_loss}, end_loss: {end_loss}")
                total_loss = torch.tensor(1.0, device=total_loss.device, requires_grad=True)
            
            return total_loss
        else:
            start_pre = torch.argmax(F.softmax(start_logits, -1), dim=-1)
            end_pre = torch.argmax(F.softmax(end_logits, -1), dim=-1)
            return start_pre, end_pre


if __name__ == '__main__':
    from transformers import DebertaConfig
    import utils
    import os

    params = utils.Params()
    # Prepare model
    deberta_config = DebertaConfig.from_json_file(os.path.join(params.bert_model_dir, 'bert_config.json'))
    model = DeBertaQueryNER.from_pretrained(config=deberta_config, pretrained_model_name_or_path=params.bert_model_dir,
                                         params=params)
    # 保存bert config
    model.to(params.device)

    for n, _ in model.named_parameters():
        print(n)
