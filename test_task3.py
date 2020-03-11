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
import random
from BertTask3.BertTask3 import BertTask3
from sklearn import metrics
import os
import re

do_test=True
all_test_dir="./PaperDataset/subtask_3/" #测试集路径
output_dir="./task3_pred_mode_2/" #训练好的模型路径
bert_model_dir="./pretrained_model/bert-base-uncased/" #bert_model模型
pred_dir="./PaperDataset/sub_task_3_pred/" #输出路径
#最多13个实体 最大token=416  最大spo:6对  关系：['Direct-Defines', 'Supplements', 'Indirect-Defines', 'Refers-To', 'AKA']

#超参数
max_seq_length=420
max_entity=14
max_spo=6
num_relation=6

relation2map={0:'None',1:'Direct-Defines',2:'Supplements', 3:'Indirect-Defines',
              4:'Refers-To', 5:'AKA'}


device=torch.device("cuda:0")


output_model_file = os.path.join(output_dir,  WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

tokenizer=BertTokenizer("./pretrained_model/bert-base-uncased-vocab.txt")

def get_pred_entity_relation_index(logits):
    '''
    logits:
        o2s_pred:[batch,max_entity,max_entity]
        os2r_pred:[batch,max_entity,num_relation]
    return:
        entity_indexs,relation_indexs=[batch,max_entity],[batch,max_entity]
    '''
    o2s_pred, os2r_pred = logits
    batch = o2s_pred.shape[0]

    entity_indexs = torch.zeros([batch, max_entity])
    relation_indexs = torch.zeros([batch, max_entity])

    for i in range(batch):
        for j in range(max_entity): #第j个实体的情况
            o2s=o2s_pred[i,j,:]
            os2r=os2r_pred[i,j,:]
            if int(o2s.argmax().item())==j:
                entity_indexs[i, j] = max_entity-1
                relation_indexs[i, j] = 0
            entity_indexs[i,j]=int(o2s.argmax().item())
            relation_indexs[i,j]=int(os2r.argmax().item())

    return [entity_indexs, relation_indexs]


def get_new_entity_info(bert_tokens, old_tag, old_tag_id):
    '''
    Args:
        bert_tokens:
        old_tag:
        old_tag_id:
    Returns:
        entity_loc #分词之后的loc
    '''
    entity_loc2tag_id = {}
    entity_loc = []
    index_old_tag = -1  # 如果old_tag.startwith("B") 代表遇见新的实体
    start = -1
    end = -1
    new_tag_id = ""
    for i, bert_token in enumerate(bert_tokens): #训练的时候可能这里的代码写错了
        if bert_token.startswith("##"):
            if start!=-1 and end!=-1:
                end=i+1
        else:
            index_old_tag += 1
            if old_tag[index_old_tag].startswith("B-"):
                if start!=-1 : #总结
                    entity_loc.append([start, end])
                    entity_loc2tag_id[str([start + 1, end + 1])] = new_tag_id
                new_tag_id = old_tag_id[index_old_tag]
                start = i
                end = i+1
            elif old_tag[index_old_tag].startswith("O"): #进行entity_loc总结
                if start != -1 and end != -1:
                    entity_loc.append([start, end])
                    entity_loc2tag_id[str([start + 1, end + 1])] = new_tag_id
                    start = -1
                    end = -1
            elif old_tag[index_old_tag].startswith("I-"):
                end = i + 1
    #测试是否检测出全部的实体
    tag_string="".join(old_tag).split("B-")
    if (len(tag_string)-1)!=len(entity_loc2tag_id):
        print()



    bert_tokens = ["[CLS]"] + bert_tokens + ["[SEP]"]
    for i in range(len(entity_loc)):
        entity_loc[i][0] += 1
        entity_loc[i][1] += 1

    return [bert_tokens, entity_loc, entity_loc2tag_id]


def removePunctuation(text):
    text = ''.join(e for e in text if e.isalnum())
    return text.strip()


def get_origin_test_json_data(origin_test_file):
    json_datas = []
    count = 0
    with open(origin_test_file, "r", encoding="utf-8") as f:
        all_texts = "".join(f.readlines())
        all_examples = all_texts.split("\n\n\n")
        for i, examlpe in enumerate(all_examples):
            if examlpe=="":
                continue
            json_datas.append({"id": count, "file_name": "", "sentence_loc": [], "tokens": [], "tag": [], "tag_id": [],
                               "token_loc": []})
            count += 1
            token_examples = examlpe.split("\n")  # 每一个token的标注情况

            for j, token_example in enumerate(token_examples):
                if token_example == "":  # 代表一句话结束
                    json_datas[-1]["sentence_loc"].append(len(json_datas[-1]["tokens"]))
                    continue
                token_examples_list = token_example.split("\t")
                json_datas[-1]["token_loc"].append([token_examples_list[2], token_examples_list[3]])
                json_datas[-1]["tokens"].append(token_examples_list[0].lstrip(" "))
                json_datas[-1]["file_name"] = token_examples_list[1]
                tag = token_examples_list[4].lstrip(" ")
                json_datas[-1]["tag"].append(tag)
                json_datas[-1]["tag_id"].append(token_examples_list[5].lstrip(" "))

    for i, example in enumerate(json_datas):

        for j, token in enumerate(example["tokens"]):

            if len(example["tokens"][j]) > 1:
                example["tokens"][j] = removePunctuation(example["tokens"][j])

        sentence = " ".join(example["tokens"])
        sentence_tokens = tokenizer.tokenize(sentence)  # 进行分词

        bert_tokens, entity_loc ,entity_loc2tag_id =\
            get_new_entity_info(sentence_tokens, example["tag"], example["tag_id"])

        json_datas[i]["bert_tokens"] = bert_tokens
        json_datas[i]["bert_entity_loc"] = entity_loc
        json_datas[i]["entity_loc2tag_id"]=entity_loc2tag_id
    return json_datas


def getTestDataTensor(json_datas):
    '''
    json_datas: 是一个文件的json_data数据,一个样本作为一个batch
    return:
    token_ids:[batch=1,max_seq]
    mask_token_ids:[batch=1,max_seq]
    mask_entity:[batch=1,max_entity]
    token_type_ids:[batch=1,max_entity]
    entity_loc:[batch=1,max_entity,2]
    '''
    token_ids = []
    mask_token_ids = []
    mask_entity = []
    token_type_ids = []
    entity_locs = []#[batch=1,max_entity,2]
    for i, example in enumerate(json_datas):
        bert_tokens = example["bert_tokens"]
        token_id = tokenizer.convert_tokens_to_ids(bert_tokens)
        assert len(bert_tokens) == len(token_id)
        token_id = torch.cat([torch.tensor(token_id).long(), torch.zeros(max_seq_length - len(token_id)).long()], dim=-1)
        mask_token_ids.append(
            torch.cat([torch.ones(len(bert_tokens)), torch.zeros(max_seq_length - len(bert_tokens))], dim=-1))
        token_ids.append(token_id)

        #example["bert_entity_loc"] [num_entity,2]
        entity_loc=torch.cat(
            [torch.tensor(example["bert_entity_loc"]).long(),
             torch.tensor([[max_seq_length-1,max_seq_length]]*(max_entity-len(example["bert_entity_loc"]))).long()],dim=0)
        entity_locs.append(entity_loc)


        mask_entity.append(
            torch.cat([torch.ones(len(example["bert_entity_loc"])),
                       torch.zeros(max_entity - len(example["bert_entity_loc"]))], dim=-1))

        token_type_ids.append(torch.zeros(max_seq_length))
    token_ids = torch.stack(token_ids).to(device)
    mask_token_ids = torch.stack(mask_token_ids).to(device)
    mask_entity = torch.stack(mask_entity).to(device)
    token_type_ids = torch.stack(token_type_ids).to(device)
    entity_locs = torch.stack(entity_locs).to(device)
    return [token_ids, mask_token_ids, mask_entity, token_type_ids, entity_locs]


def post_relu(spo, json_data):
    '''
    spo:{objtec_entity_index:[subject_entity_index,relation_index]}
    json_data:{} 只包含一个样本
    return:
    pred_json_data:
    {tokens:[],file_name:"",token_loc:[],"tag":"","tag_id":"","root_id":"","relation":""}
    '''
    pred_json_data={"tokens":[],"sentence_loc":[],"file_name":"","token_loc":[],"tag":[],"tag_id":[],"root_id":[],"relation":[]}

    pred_json_data["tokens"]=json_data["tokens"]
    pred_json_data["file_name"]=json_data["file_name"]
    pred_json_data["token_loc"]=json_data["token_loc"]
    pred_json_data["sentence_loc"]=json_data["sentence_loc"]
    pred_json_data["tag"]=json_data["tag"]
    pred_json_data["tag_id"] = json_data["tag_id"]
    object_entity_index=-1 #代表实体位置
    for i in range(len(json_data["tokens"])): #逐字赋值
        if json_data["tag"][i].startswith("O"):
            pred_json_data["root_id"].append("-1")
            pred_json_data["relation"].append("0")
        elif json_data["tag"][i].startswith("B-"):
            object_entity_index+=1

            subject_entity_index,relation_index=spo[object_entity_index]
            root_id=0
            relation=0

            if subject_entity_index==max_entity-1 or \
                    subject_entity_index>=len(json_data["bert_entity_loc"]): #无subject
                root_id=0
            else:
                root_id = json_data["entity_loc2tag_id"][str(json_data["bert_entity_loc"][subject_entity_index])]

            if relation_index==0: #无关系
                relation=0
            else:
                relation=relation2map[relation_index]

            pred_json_data["root_id"].append(root_id)
            pred_json_data["relation"].append(relation)

        elif json_data["tag"][i].startswith("I-"):
            root_id=pred_json_data["root_id"][-1]
            relation=pred_json_data["relation"][-1]
            pred_json_data["root_id"].append(root_id)
            pred_json_data["relation"].append(relation)

    return pred_json_data


def test():
    model = BertTask3.from_pretrained(bert_model_dir, num_labels=len(relation2map), max_seq_length=max_seq_length,
                                      max_entity=max_entity)
    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    model.eval()

    for test_name in os.listdir(all_test_dir):  # 遍历test文档
        print(test_name)
        origin_test_file = all_test_dir + test_name
        json_datas = get_origin_test_json_data(origin_test_file) #获取文档中的数据，并转成json文件
        pred_json_datas = []
        for json_data in json_datas:  # 遍历文档中的样本
            spo = {}  # 神经网络的预测结果 {objtec_entity_index:[subject_entity_index,relation_index]}
            if len(json_data["bert_entity_loc"]) > 0:  # 实体数量大于0时，使用神经网络预测
                token_ids, mask_token_ids, mask_entity, token_type_ids, entity_locs = getTestDataTensor(json_datas)
                with torch.no_grad():  # torch.no_grad()对于不需要反向传播的情景,禁止backward
                    # [batch=1,max_entity,max_entity,num_relation]
                    o2s_pred,os2r_pred = model(token_ids, mask_token_ids, mask_entity, token_type_ids, entity_locs, device,
                                   model_type="dev", o2s_label=None)

                #[batch,max_entity]
                entity_indexs,relation_indexs  = get_pred_entity_relation_index([o2s_pred,os2r_pred])  # [batch=1,max_entity]

                for i, object in enumerate(json_data["bert_entity_loc"]):
                    spo[i] = [int(entity_indexs[0, i]), int(relation_indexs[0, i])]
            json_data_pred=post_relu(spo, json_data)

            with open(pred_dir+test_name,"a",encoding="utf-8") as f:
                for i in range(len(json_data_pred["tokens"])):
                    if i in json_data_pred["sentence_loc"]:
                        print("\n",file=f)
                    print(json_data_pred["tokens"][i],"\t",json_data_pred["file_name"],"\t",json_data_pred["token_loc"][i][0],"\t",json_data_pred["token_loc"][i][1],"\t",json_data_pred["tag"][i],"\t",json_data_pred["tag_id"][i],"\t",json_data_pred["root_id"][i],"\t",json_data_pred["relation"][i],file=f)
                print("\n\n",file=f)



test()