#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import pkg_resources
import time
import scipy.stats as stats
import gc
import re
import operator
import sys
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold,KFold
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from nltk.stem import PorterStemmer
from sklearn.metrics import roc_auc_score
from torch.nn.parallel.data_parallel import data_parallel
from apex.parallel import DistributedDataParallel
from torch.utils.data.sampler import *
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert.modeling import BertEmbeddings, BertEncoder, BertConfig,  BertPreTrainedModel
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--gpu", default='1', type=str)
parser.add_argument("--fold", default=0, type=int)
parser.add_argument("--train", default=1, type=int)
parser.add_argument("--path", default=None, type=str)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_split = args.fold
TRAIN = bool(args.train)
weight_name = args.path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

debug = False

MAX_SEQUENCE_LENGTH = 220
SEED = 1314112342
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input/jigsaw-unintended-bias-in-toxicity-classification"
WORK_DIR = "output"


if debug:
    train_size = 500                        #Train size to match time limit
    valid_size = 500                          #Validation Size
    
else:
    train_size = 1600000                        #Train size to match time limit
    valid_size = 200000 
    
num_to_load = train_size
valid_size = valid_size
TOXICITY_COLUMN = 'target'
bert_loc = 'bert-base-uncased'


# # Evaluate functions

# From baseline kernel
def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]>0.5
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
                             power_mean(bias_df[SUBGROUP_AUC], POWER),
                             power_mean(bias_df[BPSN_AUC], POWER),
                             power_mean(bias_df[BNSP_AUC], POWER)
                             ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)



SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]>0.5]
    return compute_auc((subgroup_examples[label]>0.5), subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[(df[subgroup]>0.5) & (df[label]<=0.5)]
    non_subgroup_positive_examples = df[(df[subgroup]<=0.5) & (df[label]>0.5)]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[(df[subgroup]>0.5) & (df[label]>0.5)]
    non_subgroup_negative_examples = df[(df[subgroup]<=0.5) & (df[label]<=0.5)]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label]>0.5, examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]>0.5])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)

################################################################################################

# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming

truncate_options = ['head', 
                    'tail', 
                    'both']

truncate_option = truncate_options[2]

def convert_lines(example, max_seq_length,tokenizer, truncate_option=truncate_option):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        
        tokens_a = tokenizer.tokenize(text)
        
        if len(tokens_a)>max_seq_length:
            
            if truncate_option == 'head':
                
                tokens_a = tokens_a[:max_seq_length]
            
            elif truncate_option == 'tail':
                
                tokens_a = tokens_a[-max_seq_length:]

            elif truncate_option == 'both':
                
                if len(tokens_a) > max_seq_length:
                    
                    tokens_a = tokens_a[:int(max_seq_length/2)] + tokens_a[-int(max_seq_length/2):]
                    
                
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased/bert-base-uncased-vocab.txt, 
                                          cache_dir=None,
                                          do_lower_case=True)

train_df = pd.read_csv(os.path.join(Data_dir,"train.csv"))
print('loaded %d records' % len(train_df))

# Make sure all comment_text values are strings
train_df['comment_text'] = train_df['comment_text'].astype(str) 
if not os.path.exists(os.path.join(WORK_DIR,"sequences.train.pkl")) or debug:
    sequences = convert_lines(train_df["comment_text"].fillna("DUMMY_VALUE"),MAX_SEQUENCE_LENGTH,tokenizer)
    if not debug:
        pd.to_pickle(sequences,os.path.join(WORK_DIR,"sequences.train.pkl"))
else:
    sequences = pd.read_pickle(os.path.join(WORK_DIR,"sequences.train.pkl"))
train_df=train_df.fillna(0)

################################################################################################


# List all identities
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']

train_df = train_df.drop(['comment_text'],axis=1)
# convert target to 0,1
#---------------------- 0602
train_df['target_binary']=(train_df['target']>=0.5).astype(float)
y_columns = ['target_binary'] + ['target',
                                 'severe_toxicity',
                                 'obscene',
                                 'identity_attack',
                                 'insult',
                                 'threat',
                                 # 'sexual_explicit'
                                ] # + identity_columns
#-------------------------------------
skf = KFold(n_splits=10, shuffle=True,random_state=SEED).split(sequences,train_df[y_columns].values)
for split,(train_idx,test_idx) in enumerate(skf):
    if split == use_split:
        print(len(train_idx),len(test_idx))
        X = sequences[train_idx]                
        y = train_df[y_columns].values[train_idx]
        X_val = sequences[test_idx]                
        y_val = train_df[y_columns].values[test_idx]
        break

#X = sequences[:num_to_load]                
#y = train_df[y_columns].values[:num_to_load]
#X_val = sequences[num_to_load:]                
#y_val = train_df[y_columns].values[num_to_load:]


#test_df=train_df.tail(valid_size).copy()
#train_df=train_df.head(num_to_load)
test_df = train_df.iloc[test_idx].copy()
train_df = train_df.iloc[train_idx]

#---------------- 0601
# Overall
weights = np.ones((len(train_df),)) / 4

# Subgroup
weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4

# Background Positive
weights += (( (train_df['target'].values>=0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4


# Subgroup Negative
weights += (( (train_df['target'].values<0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4


# loss_weight = 1.0 / weights.mean() 
# weights = weights * loss_weight 
#----------------

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), 
                                               torch.tensor(y,dtype=torch.float),
                                               torch.tensor(weights,dtype=torch.float)
                                              )
valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.long), torch.tensor(y_val, dtype=torch.float))

lr=2e-5
batch_size = 32*1
accumulation_steps = 1
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


################################################################################################

class BertClassification(BertPreTrainedModel):
    

    def __init__(self, config, num_labels=2):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        
        #-------------------- dense in new pooler 
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        #--------------------
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, 
                                                  output_all_encoded_layers=True)
        
        #--------------- use another pooling
        hidden_states = encoded_layers[-1]

        #--- lase element pooling
        sent_embed = hidden_states[:, 0]
        
        pooled_output = self.dense(sent_embed)
        pooled_output = self.activation(pooled_output)
        
        #---------------
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


# In[6]:


################################################################################################

# model = BertForSequenceClassification.from_pretrained(bert_loc,
#                                                       cache_dir=None,
#                                                       num_labels=len(y_columns))
# torch.cuda.set_device(0)
# torch.distributed.init_process_group(backend='nccl',init_method='env://')
# torch.backends.cudnn.benchmark = True
model = BertClassification.from_pretrained('bert-base-uncased/',
                                          cache_dir=None,
                                          num_labels=len(y_columns)).cuda()


model.zero_grad()
param_optimizer = list(model.named_parameters())

for param in model.bert.embeddings.parameters():
    param.requires_grad = False

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

train = train_dataset
valid = valid_dataset

num_train_optimization_steps = int(EPOCHS*len(train)/batch_size/accumulation_steps)

optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=0.05,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)
#model = DistributedDataParallel(model).cuda()
#model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
#model = torch.nn.parallel.DistributedDataParallel(model,
#                                                  device_ids=[0,1,2,3],
#                                                  output_device=0).cuda()
tq = tqdm(range(EPOCHS))


for epoch in tq:
    if TRAIN:
        model.train()
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        avg_loss = 0.
        avg_accuracy = 0.
        lossf=None
        tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        optimizer.zero_grad()   # Bug fix - thanks to @chinhuic
        for i,(x_batch, y_batch, w_batch) in tk0:

            y_pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda(), labels=None)

            y_batch = y_batch.cuda()
            w_batch = w_batch.cuda()


            loss_main =  F.binary_cross_entropy_with_logits(input=y_pred[:,0],
                                                       target=y_batch[:,0],
                                                       weight=w_batch)

            loss_aux =  F.binary_cross_entropy_with_logits(input=y_pred[:, 1:],
                                                           target=y_batch[:,1:])

            loss = loss_main + loss_aux * 6

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            # loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
        tq.set_postfix(avg_loss=avg_loss)

    ################################################################################################
    ### validation
    model.eval()
    if not TRAIN:
        print("load weight...")
        model.load_state_dict(torch.load("output/{}".format(weight_name)))
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=32, shuffle=False)
    valid_preds = np.zeros((len(X_val)))
    tk1 = tqdm(valid_loader)
    valid_accuracy = 0.0
    
    for i,(x_batch, y_batch)  in enumerate(tk1):

        pred = model(x_batch.cuda(), attention_mask=(x_batch>0).cuda(), labels=None).detach()[:,0]
        valid_preds[i*batch_size:(i+1)*batch_size]=(torch.sigmoid(pred[:])).cpu().squeeze().detach().numpy()

    valid_auc = roc_auc_score(y_val[:,0], valid_preds[:])
    
    print("valid_auc: ", valid_auc)
    
    MODEL_NAME = 'BERT'
    test_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()
    TOXICITY_COLUMN = 'target_binary'
    bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
    print(bias_metrics_df)
    
    final_metric = get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME))
    print('final metric is', final_metric)
    
    if TRAIN:
        output_model_file = 'output/BERT_fold_{}_epoch_{}_fix_embed_dopout0.1_bs32_{}_{:0.5f}.bin'.format(use_split,epoch, 
                                                                                              truncate_option,
                                                                                              final_metric)
        torch.save(model.state_dict(), output_model_file)
    
    SEED += 1
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    ################################################################################################