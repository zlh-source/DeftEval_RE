from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import numpy as np


max_entity = 14 #13号实体代表不对应任何实体
num_head=1

def get_o2s_mask(mask_entity,device):
    '''
    mask_entity:[batch,max_entity]
    return:
    o2s_mask:[batch,max_entity,max_entity]
    '''
    o2s_mask = torch.full([mask_entity.shape[0], max_entity, max_entity],-100.0).to(device)
    for i in range(mask_entity.shape[0]):
        n = int(torch.sum(mask_entity[i]).item())
        o2s_mask[i, :n, :n] = 0
    o2s_mask[:,:,max_entity-1]=0
    o2s_mask[:, max_entity - 1, max_entity - 1] = 100.0
    return o2s_mask

class BertTask3(BertPreTrainedModel):
    '''
    转化成序列标注问题，序列长度为 num_label*max_seq
    '''

    def __init__(self, config, num_labels, max_seq_length, max_entity):
        super(BertTask3, self).__init__(config)
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.max_entity = max_entity
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self.init_bert_weights)

        self.bateLinear = nn.Linear(2, max_seq_length)  # 2->420
        # nn.init.constant(self.bateLinear,torch.zeros(max_seq_length,2))
        self.gammaLineas = nn.Linear(2, max_seq_length)  # 2->420
        # nn.init.constant(self.gammaLineas, torch.zeros(max_seq_length, 2))
        self.linear=nn.Linear(max_seq_length,max_entity) #420->13
        self.normal=nn.LayerNorm(normalized_shape=[max_entity,max_seq_length])
        self.linear2=nn.Linear(2*config.hidden_size,num_labels)

    def forward(self, token_ids, mask_token_ids, mask_entity, token_type_ids, entity_loc, device,model_type="train",o2s_label=None):
        '''
        Args:
            token_ids: [batch,max_seq_length]
            mask_token_ids:  [batch,max_seq_length]
            mask_entity: [batch,max_entity]
            token_type_ids: [batch,max_seq_length]
            entity_loc: [batch,max_entity,2]
            o2s_label:[batch,max_entity,max_entity] 表示os之间的真正对应关系
        Returns:
            o2s_pred: [batch,max_entity,max_entity]
            os2r_pred: [batch,max_entity,num_label]
        '''
        encoded_layers, _ = self.bert(  # [batch,max_seq_legth,bert_dim]
            token_ids.long(), token_type_ids.long(), mask_token_ids.long(), output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)

        subject = []  # [batch,max_entity,2,bert_dim]
        # 抽出subject首尾对应的编码向量
        for i in range(token_ids.shape[0]):  # batch=8
            subject.append([])
            for j in range(max_entity):  # 12
                start, end = entity_loc[i, j, :]
                s1 = encoded_layers[i][int(start.item())].unsqueeze(dim=0)
                s2 = encoded_layers[i][int(end.item()) - 1].unsqueeze(dim=0)
                subject[-1].append(torch.cat([s1, s2], dim=0).to(device))
            subject[-1] = torch.stack(subject[-1]).to(device)
        all_subject = torch.stack(subject).to(device)  # [batch,max_entity,2,bert_dim]


        # encoded_layers:[batch,all_seq,dim]  subject:[batch,max_entity,2,dim]
        subject_output = []
        # LayerNormalization
        encoded_layers = encoded_layers.transpose(-1, -2)  # [batch,bert_dim,max_seq]
        for i in range(max_entity):  # 对每个实体进行CLN
            subject = all_subject[:, i, :, :].transpose(-1, -2)  # [batch,bert_dim,2]
            beta = torch.zeros(self.max_seq_length).to(device) + self.bateLinear(subject)  # [batch,bert_dim,max_seq]
            gamma = torch.ones(self.max_seq_length).to(device) + self.gammaLineas(subject)  # [batch,bert_dim,max_seq]
            # bate:[batch,cond_seq,max_seq_length]  gama:[batch,cond_seq,max_seq_length]
            mean = torch.mean(encoded_layers, dim=-1, keepdim=True).to(device)  # [batch,bert_dim,1]
            variance = torch.mean(torch.pow(encoded_layers - mean, 2), dim=-1, keepdim=True).to(device)
            std = torch.sqrt(variance + (1e-8) * (1e-8)).to(device)
            outputs = (encoded_layers - mean) / std
            outputs = outputs * gamma + beta  # [batch,dim,max_seq]
            outputs = outputs.transpose(-1, -2)  # [batch,max_seq,bert_dim]  条件融合之后的文本表示
            subject_output.append(outputs) #[max_entity,batch,max_seq,bert_dim]

        subject_output=torch.stack(subject_output).transpose(0,1).to(device) #[batch,max_entity,max_seq,dim]
        subject_output=self.normal(subject_output.sum(dim=-1)) #[batch,max_entity,max_seq]
        o2s_pred=self.linear(subject_output) #[batch,max_entity,max_entity]

        all_subject=all_subject.reshape(all_subject.shape[0],all_subject.shape[1],-1) #[batch,max_entity,2*bert_dim]
        all_subject=self.linear2(all_subject) #[batch,max_entity,num_label]

        os2r_pred=torch.matmul(o2s_pred,all_subject).to(device) #[batch,max_entity,num_label]

        return [o2s_pred,os2r_pred] #[batch,max_entity,max_entity]  #[batch,max_entity,num_label]