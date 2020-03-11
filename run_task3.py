import os
import numpy as np
import json
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset,SequentialSampler)
import torch.nn as nn
import torch
from tqdm import tqdm
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME
import random
from BertTask3.BertTask3 import BertTask3
from sklearn import metrics
import os

do_test=True
train_path="train.json"
dev_path="dev.json"
output_dir="./task3_pred_mode_4/" #输出的最好的模型 #注意不要覆盖原文件
bert_model_dir="./pretrained_model/bert-large-cased/"
task3_log_path="task3_9.txt"

info="large_bert关系损失 不加权重 ，无mask，分成两个损失"

#最多13个实体 最大token=416  最大spo:6对  关系：['Direct-Defines', 'Supplements', 'Indirect-Defines', 'Refers-To', 'AKA']

#超参数
max_seq_length=420
max_entity=14 #13号实体代表不对应
max_spo=6
num_relation=6
num_train_epochs=20.0

relation2map={'None':0,'Direct-Defines':1, 'Supplements':2, 'Indirect-Defines':3,
              'Refers-To':4, 'AKA':5}

train_batch_size=64
dev_batch_size=64
gradient_accumulation_steps=32
seed=0
do_train=True
pow_n=1  #苏简琳 类别不均衡的问题
learning_rate=5e-5
warmup_proportion=0.1
print_step=20
device=torch.device("cuda:3")
#gpu_id=3

train_batch_size =train_batch_size //gradient_accumulation_steps
output_model_file = os.path.join(output_dir,  WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

tokenizer=BertTokenizer("./pretrained_model/bert-large-cased-vocab.txt")
criterion_entity = nn.CrossEntropyLoss()

criterion_relation = nn.CrossEntropyLoss()


def summer(a):
    # count=[0]*6
    # all=int(a.sum().item())
    # d1,d2,d3=a.shape
    # for i in range(d1):
    #     for j in range(d2):
    #         for k in range(d3):
    #             item=int(a[i,j,k].item())
    #             count[item]+=1
    # print(count)
    # for i in count:
    #     print(10.0*all/i,end=" , ")
    # print("over")
    return


def get_dataloader(mode_type):
    path=train_path if mode_type=="train" else dev_path
    batch_size=train_batch_size if mode_type=="train" else dev_batch_size
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    all_token_ids=[] #[batch,max_seq]
    all_mask_token_ids=[]  #[batch,max_seq]
    all_mask_entity=[]  #[batch,max_entity]
    all_entity_loc=[]  #[batch,max_entity,2]
    all_num_entity=[] #[batch]
    all_o2s_label=[] #[batch,max_entity]
    all_os2r_label=[] #[batch,max_entity]

    # num_relations=[0]*num_relation
    # sum_entity=0
    # sum_spo=0
    for i,example in tqdm(enumerate(json_data)):
        # sum_spo+=len(example["spo_list"])
        # sum_entity+=len(example["entity"])

        if len(example["entity"])==0: #跳过没有实体的样本,没有实体，用规则即可
            continue

        tokens2id=tokenizer.convert_tokens_to_ids(example["tokens"])

        #all_token_ids:[batch,max_seq]
        all_token_ids.append(torch.tensor(tokens2id + [0.0] * (max_seq_length - len(tokens2id))))

        #all_mask_token_ids:[batch,max_seq]
        all_mask_token_ids.append(
            torch.tensor([1.0] * len(example["tokens"]) + [0.0] * (max_seq_length - len(example["tokens"]))))

        #all_num_entity:[batch]
        all_num_entity.append(len(example["entity"]))

        #all_o2s_label:[batch,max_entity]
        all_o2s_label.append(torch.full([max_entity],max_entity-1))
        all_os2r_label.append(torch.zeros([max_entity]))
        for spo in example["spo_list"]:
            o_index=example["entity"].index(spo["object"])
            s_index=example["entity"].index(spo["subject"])
            r_index=relation2map[spo["relation"]]
            all_o2s_label[-1][o_index]=s_index
            all_os2r_label[-1][o_index]=r_index

        #all_entity_loc:[batch,max_entity,2]
        all_entity_loc.append([])#添加新的batch
        for j,entity in enumerate(example["entity"]):
            start,end=entity
            all_entity_loc[-1].append(torch.tensor([start,end]))
        for j in range(max_entity-len(example["entity"])):
            all_entity_loc[-1].append(torch.tensor([max_seq_length-1,max_seq_length]))
        all_entity_loc[-1]=torch.stack(all_entity_loc[-1])


        #all_mask_entity (batch,max_entity)
        all_mask_entity.append(
            torch.cat([torch.ones(len(example["entity"])),torch.zeros(max_entity-len(example["entity"]))],dim=-1))

    # num_relations[0]=sum_entity-sum_spo  #总实体数-总的spo关系
    # print(num_relations)
    # weight=[]
    # for num in num_relations:
    #     weight.append(1.0*sum(num_relations)/num)
    # for num in weight:
    #     print(num/sum(weight))


    all_os2r_label=torch.stack(all_os2r_label) #[batch,max_entity]
    all_o2s_label=torch.stack(all_o2s_label) #[batch,max_entity]
    all_token_ids=torch.stack(all_token_ids) # [batch,max_seq]
    all_token_type_ids = torch.zeros(all_token_ids.shape) #[batch,max_seq]
    all_mask_token_ids=torch.stack(all_mask_token_ids) #[batch,max_seq]
    all_mask_entity=torch.stack(all_mask_entity)  #[batch,max_entity]
    all_entity_loc=torch.stack(all_entity_loc) #[batch,max_entity,2]
    all_num_entity=torch.tensor(all_num_entity) #[batch]

    dataset=TensorDataset(all_token_ids,all_o2s_label,all_os2r_label,all_mask_token_ids,all_mask_entity,all_token_type_ids,all_entity_loc,all_num_entity)

    if mode_type=="train":
        sampler=RandomSampler(dataset)
    else:
        sampler=SequentialSampler(dataset)

    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)

    return dataloader,len(json_data)


def get_pred_entity_relation_index(logits,entity_mask):
    '''
    logits:
        o2s_pred:[batch,max_entity,max_entity]
        os2r_pred:[batch,max_entity,num_label]
        entity_mask:[batch,mex_entity]
    return:
        entity_indexs:[batch*num_entity]
        relation_indexs:[batch*num_entity]
    '''
    o2s_pred,os2r_pred=logits

    batch=o2s_pred.shape[0]

    entity_indexs = []
    relation_indexs = []

    for i in range(batch):
        for j in range(int(torch.sum(entity_mask[i]).item())):
            o2s=o2s_pred[i,j,:] #[max_entity]
            os2r=os2r_pred[i,j,:] #[num_label]
            entity_indexs.append(int(o2s.argmax().item()))
            relation_indexs.append(int(os2r.argmax().item()))
    return [entity_indexs,relation_indexs]


def get_label_entity_relation_index(label,entity_mask):
    '''
    label:
        o2s_label:[batch,max_entity]
        os2r_label:[batch,max_entity]
        entity_mask:[batch,mex_entity]
    return:
        entity_indexs:[batch*num_entity]
        relation_indexs:[batch*num_entity]
    '''
    o2s_label,os2r_label=label
    entity_indexs, relation_indexs=[],[]

    batch=o2s_label.shape[0]
    for i in range(batch):
        for j in range(int(torch.sum(entity_mask[i,:]))):
            entity_indexs.append(int(o2s_label[i,j].item()))
            relation_indexs.append(int(os2r_label[i,j].item()))
    return [entity_indexs,relation_indexs]

def evaluate(epoch,model,dev_data_loader):

    all_rootid_preds = np.array([], dtype=int)
    all_rootid_labels = np.array([], dtype=int)
    all_relation_preds = np.array([], dtype=int)
    all_relation_labels = np.array([], dtype=int)

    epoch_loss = 0

    for token_ids,o2s_label,os2r_label,mask_token_ids,mask_entity,token_type_ids,entity_loc,num_entity  in tqdm(dev_data_loader, desc="Eval",ncols=100):
        '''
        token_ids:[batch,max_seq]
        mask_token_ids:[batch,max_seq]
        mask_entity:[batch,max_entity]
        token_type_ids:[batch,max_seq]
        entity_loc:[batch,max_entity,2]
        num_entity:[batch]
        o2s_relation_label:[batch,max_entity,max_entity]
        '''
        model.eval()
        token_ids, o2s_label,os2r_label, mask_token_ids, mask_entity, token_type_ids, entity_loc, num_entity = token_ids.to(
            device), o2s_label.to(device),os2r_label.to(device), mask_token_ids.to(device), mask_entity.to(device), token_type_ids.to(
            device), entity_loc.to(device), num_entity.to(device)

        with torch.no_grad(): #torch.no_grad()对于不需要反向传播的情景,禁止backward
            # [batch,max_entity,max_entity]  [batch,max_entity,num_label]
            o2s_pred,os2r_pred = model(token_ids, mask_token_ids, mask_entity, token_type_ids, entity_loc, device,model_type="dev",o2s_label=None)

        # logots_entity_index,logits_relation_index:[batch,max_entity]
        # label_entity_index, label_relation_index:[batch,max_entity]


        logits_entity_index,logits_relation_index=get_pred_entity_relation_index([o2s_pred.detach().cpu(),os2r_pred.detach().cpu()],mask_entity.to("cpu"))
        label_entity_index, label_relation_index = get_label_entity_relation_index([o2s_label.to("cpu"),os2r_label.to("cpu")],mask_entity.to("cpu"))


        all_rootid_preds = np.hstack([all_rootid_preds,np.array(logits_entity_index)])
        all_rootid_labels = np.hstack([all_rootid_labels,np.array(label_entity_index)])
        all_relation_preds = np.hstack([all_relation_preds,np.array(logits_relation_index)])
        all_relation_labels = np.hstack([all_relation_labels,np.array(label_relation_index)])


    #all_labels:[batch*max_entity*max_entity]
    acc_rootid=metrics.accuracy_score(all_rootid_labels,all_rootid_preds)
    f1_rootid=metrics.f1_score(all_rootid_labels,all_rootid_preds,average="macro")
    acc_relation=metrics.accuracy_score(all_relation_labels,all_relation_preds)
    f1_relation=metrics.f1_score(all_relation_labels,all_relation_preds,average="macro")

    print()
    print("acc_rootid:",acc_rootid,"macro_f1_root_id:",f1_rootid)
    print("acc_relation:",acc_relation,"macro_f1_relation:",f1_relation)
    print()

    with open(task3_log_path,"a",encoding="utf-8") as f:
        print("info:",info,"epoch:",epoch,
              "acc_rootid:",acc_rootid,"macro_f1_root_id:",f1_rootid,
              "acc_relation:",acc_relation,"macro_f1_relation:",f1_relation,file=f)

        print("preds_rootid","\t","label_rootid","\t","preds_relation","\t","label_relation", file=f)
        for i,j,k,w in zip(all_rootid_preds,all_rootid_labels,all_relation_preds,all_relation_labels):
            print(i,"\t",j,"\t",k,"\t",w,file=f)
        print("\n",file=f)

    return  [acc_rootid,f1_rootid,acc_relation,f1_relation]


def main():
    #设置随机数种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if do_train:
        train_data_loader,train_examples_len=get_dataloader(mode_type="train")

        dev_data_loader ,_= get_dataloader(mode_type="dev")

        num_train_optimization_steps = int(train_examples_len/train_batch_size/gradient_accumulation_steps)*num_train_epochs

        #模型构建
        model = BertTask3.from_pretrained(bert_model_dir,num_labels=len(relation2map),max_seq_length=max_seq_length,max_entity=max_entity)

        model.to(device)

        """ 优化器准备 """
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=learning_rate,
                             warmup=warmup_proportion,
                             t_total=num_train_optimization_steps)

        global_step=0

        best_f1_relation = 0

        for epoch in range(int(num_train_epochs)):
            epoch_loss=0.0
            train_steps=0

            for step,batch in enumerate(tqdm(train_data_loader,desc="Iteration",ncols=80)):
                model.train()
                token_ids,o2s_label,os2r_label,mask_token_ids,mask_entity,token_type_ids,entity_loc,num_entity = batch
                '''
                token_ids:[batch,max_seq]
                mask_token_ids:[batch,max_seq]
                mask_entity:[batch,max_entity]
                token_type_ids:[batch,max_seq]
                entity_loc:[batch,max_entity,2]
                num_entity:[batch]
                o2s_label:[batch,max_entity]
                os2r_label:[batch,max_entity]
                '''
                token_ids, o2s_label,os2r_label, mask_token_ids, mask_entity, token_type_ids, entity_loc, num_entity= token_ids.to(device),o2s_label.to(device),os2r_label.to(device),mask_token_ids.to(device),mask_entity.to(device),token_type_ids.to(device),entity_loc.to(device),num_entity.to(device)

                #o2s_pred  [batch,max_entity,max_entity]
                #os2r_pred  [batch,max_entity,num_label]
                o2s_pred,os2r_pred=model(
                    token_ids, mask_token_ids, mask_entity, token_type_ids, entity_loc, device,model_type="train",o2s_label=None)

                #o2s_relation_pred=torch.pow(o2s_relation_pred,pow_n).to(device)

                loss_o2s=criterion_entity(o2s_pred.reshape(-1,o2s_pred.shape[-1]),o2s_label.reshape(-1).long())  #损失函数设置技巧 参考苏简琳
                loss_os2r=criterion_relation(os2r_pred.reshape(-1,os2r_pred.shape[-1]),os2r_label.reshape(-1).long())
                loss=loss_o2s+loss_os2r

                '''修正loss'''
                if gradient_accumulation_steps>1:
                    loss=loss/gradient_accumulation_steps

                train_steps += 1
                loss.backward()

                # 累计梯度到一定次数后，进行梯度下降
                if (step+1)%gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step+=1
                    if global_step % print_step==0 and global_step!=0:
                        print()
                        print(loss)
                        print()
                        _,_,_,f1_relation=evaluate(epoch,model, dev_data_loader)
                        if best_f1_relation<f1_relation:
                            best_f1_relation=f1_relation
                            # 保存模型
                            model_to_save = model.module if hasattr(
                                model, 'module') else model
                            torch.save(model_to_save.state_dict(), output_model_file)
                            with open(output_config_file, 'w') as f:
                                f.write(model_to_save.config.to_json_string())

main()