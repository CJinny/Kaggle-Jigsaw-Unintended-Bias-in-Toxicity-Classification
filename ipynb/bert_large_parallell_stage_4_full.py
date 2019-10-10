#extra_lr = 1e-6        # after 2 epochs


LR = 2e-5 #previous LR
lr=2e-5
batch_size = 96
accumulation_steps=1
OUTDIR = 'bert_large_models'

import numpy as np
import pandas as pd
from pathlib import Path
from typing import *
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
import torch.nn as nn

import torch.optim as optim
from fastai import *
from fastai.vision import *
from fastai.text import *
from fastai.callbacks import *
from tqdm import tqdm, tqdm_notebook
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings
warnings.filterwarnings(action='once')
import pickle
from apex import amp
import shutil
device=torch.device('cuda:0')

package_dir_a = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.insert(0, package_dir_a)


from pytorch_pretrained_bert import convert_tf_checkpoint_to_pytorch
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import OpenAIAdam

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3,4,5,6,7"  # specify which GPU(s) to be used

MAX_SEQUENCE_LENGTH = 220
SEED = 1234
EPOCHS = 1
Data_dir="../input/jigsaw-unintended-bias-in-toxicity-classification"
Input_dir = "../input"

#! mkdir ../working2
WORK_DIR = "../working2/"

TOXICITY_COLUMN = 'target'

#BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_L-12_H-768_A-12/'
BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_L-24_H-1024_A-16/'
#shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'bert_config.json')

from pytorch_pretrained_bert import BertConfig

bert_config = BertConfig('../input/bert-pretrained-models/uncased_L-24_H-1024_A-16/'+'bert_config.json')

def convert_lines(example, max_seq_length,tokenizer):
    max_seq_length -=2
    all_tokens = []
    longer = 0
    for text in tqdm_notebook(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a)>max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    print(longer)
    return np.array(all_tokens)

BERT_MODEL_PATH = '../input/bert-pretrained-models/uncased_L-24_H-1024_A-16/'

convert_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    BERT_MODEL_PATH + 'bert_model.ckpt',
    BERT_MODEL_PATH + 'bert_config.json',
    WORK_DIR + 'pytorch_model.bin')
shutil.copyfile(BERT_MODEL_PATH + 'bert_config.json', WORK_DIR + 'config.json')



# In[3]:


torch.cuda.device_count()
torch.cuda.get_device_name(0)
torch.cuda.get_device_properties(device).total_memory
torch.cuda.is_available()
torch.cuda.memory_allocated(device=None)
print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))

torch.cuda.empty_cache()



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


# In[5]:


print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))


# In[6]:


import pickle

train_df = pd.read_csv(os.path.join(Data_dir,"train.csv"))
print('loaded %d records' % len(train_df))
train_df['comment_text'] = train_df['comment_text'].astype(str) 
train_df=train_df.fillna(0)
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
y_columns=['target']
train_df = train_df.drop(['comment_text'],axis=1)
train_df['target']=(train_df['target']>=0.5).astype(float)

with open('skf_5_splits.pkl', 'rb') as f:
    splits = pickle.load(f)

#skf = StratifiedKFold(n_splits=5, random_state=True, shuffle=True)
#splits = list(skf.split(train_df, train_df['target']))


I = 0
WARMUP=0.05
SEED = 1234
EPOCHS = 1
sequences = np.load('jin_files/bert_large_sequences_220.npy')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)


#X = sequences[splits[I][0]]
## for full model
X = sequences
X_val = sequences[splits[I][1]]
test_df = train_df.iloc[splits[I][1],:]

weights = np.ones(len(train_df)) / 4
weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
weights += (( (train_df['target'].values >= 0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
weights += (( (train_df['target'].values < 0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4

Y = np.vstack([(train_df['target'].values >= 0.5).astype(np.int), weights]).T
Y_aux = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
Y = np.hstack((Y, Y_aux))

#y = Y[splits[I][0]]
## for full model
y = Y
y_val = Y[splits[I][1]]

del train_df, sequences, Y, Y_aux, weights, splits
gc.collect()

train_dataset = torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.long), torch.tensor(y,dtype=torch.float))
del X, y, tokenizer
gc.collect()

def custom_loss(y_pred, y_true):
    bce_loss_1 = F.binary_cross_entropy_with_logits(y_pred[:,:1].reshape(-1), y_true[:,:1].reshape(-1), weight=y_true[:,1:2].reshape(-1))   
    N = y_true.shape[1]
    custom_loss = bce_loss_1
    for i in range(N-2):
        custom_loss += F.binary_cross_entropy_with_logits(y_pred[:,1+i], y_true[:,2+i])
    return custom_loss





def generate_params(epoch_len, lr=2e-5, STAGE=1, WARMUP=0.05, SEED=1234, num_stages=4, warmup=0.05):
   
    assert STAGE <= num_stages
    if STAGE==1:
        FACTOR=1
        MIN=0
        MAX=(epoch_len*2)//num_stages
        
    else:
        FACTOR = (num_stages+1-STAGE)/num_stages
        MIN=((epoch_len*2)//num_stages) * (STAGE-1)+1
        if MIN >= epoch_len:
            MIN -= epoch_len+1
        MAX=((epoch_len*2)//num_stages) * (STAGE)
        if MAX > epoch_len:
            MAX -= epoch_len
        warmup=0
        if STAGE > num_stages/2:
            SEED = SEED*2

    lr=lr*FACTOR
    return lr, MIN, MAX, STAGE, SEED, warmup, FACTOR

epoch_len = len(train_dataset)//batch_size
print(epoch_len)

#lr, MIN, MAX, STAGE, SEED, WARMUP, FACTOR = generate_params(epoch_len, lr=lr, STAGE=1, num_stages=4)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
del train_dataset,
gc.collect()


torch.cuda.empty_cache()
print('cuda memory allocated is: {}'.format(torch.cuda.memory_allocated()))

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))




#############################################################################
#stage 1 begins
'''
print('stage 1 begins')
lr, MIN, MAX, STAGE, SEED, WARMUP, FACTOR = generate_params(epoch_len, lr=lr, STAGE=1, num_stages=4)
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)
model.zero_grad()

model = model.to(device)
#model = model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = int(FACTOR*2*EPOCHS * len(train_loader.tensors[0]) /batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=WARMUP,
                     t_total=num_train_optimization_steps)
                      
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

#######################
# multi-gpu
#######################
model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5])

model=model.train()


del param_optimizer, optimizer_grouped_parameters
gc.collect()

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tq = tqdm(range(EPOCHS))
for epoch in tq:
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    # zero.grad() moved ahead
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
        if i < MIN:
            pass
        elif i > MAX:
            pass
        
        else:
            #optimizer.zero_grad()
            y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            #loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            loss = custom_loss(y_pred, y_batch.to(device))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
            
            if i % 1000 == 0:
                print('saving model checkpoint at iteration={}'.format(i))
                torch.save(model.module.state_dict(), '{}/bert_large_fold_{}_epoch_1_ckpt_iteration_{}.bin'.format(OUTDIR, I, i))
            
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    
    
output_model_file = '{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE)
torch.save(model.module.state_dict(), output_model_file)

######################################################################

#stage 2 begins

print('stage 2 begins')
lr, MIN, MAX, STAGE, SEED, WARMUP, FACTOR = generate_params(epoch_len, lr=lr, STAGE=2, num_stages=4)
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)
#model = nn.DataParallel(model)
model.load_state_dict(torch.load('{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE-1)))
#model = model.module

model = model.to(device)
#model = model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = int(FACTOR*2*EPOCHS * len(train_loader.tensors[0]) /batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=WARMUP,
                     t_total=num_train_optimization_steps)
                      
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

#######################
# multi-gpu
#######################
model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5])

model=model.train()

del param_optimizer, optimizer_grouped_parameters
gc.collect()

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tq = tqdm(range(EPOCHS))
for epoch in tq:
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    # zero.grad() moved ahead
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
        if i < MIN:
            pass
        elif i > MAX:
            pass
        
        else:
            #optimizer.zero_grad()
            y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            #loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            loss = custom_loss(y_pred, y_batch.to(device))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
            
            
            if i % 1000 == 0:
                print('saving model checkpoint at iteration={}'.format(i))
                torch.save(model.module.state_dict(), '{}/bert_large_fold_{}_epoch_1_ckpt_iteration_{}.bin'.format(OUTDIR, I, i))
            
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    
output_model_file = '{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE)
torch.save(model.module.state_dict(), output_model_file)

######################################################################

#stage 3 begins

print('stage 3 begins')
lr, MIN, MAX, STAGE, SEED, WARMUP, FACTOR = generate_params(epoch_len, lr=lr, STAGE=3, num_stages=4)
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)
#model = nn.DataParallel(model)
model.load_state_dict(torch.load('{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE-1)))
#model = model.module
model = model.to(device)
#model = model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = int(FACTOR*2*EPOCHS * len(train_loader.tensors[0]) /batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=WARMUP,
                     t_total=num_train_optimization_steps)
                      
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

#######################
# multi-gpu
#######################
model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5])

model=model.train()

del param_optimizer, optimizer_grouped_parameters
gc.collect()

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tq = tqdm(range(EPOCHS))
for epoch in tq:
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    # zero.grad() moved ahead
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
        if i < MIN:
            pass
        elif i > MAX:
            pass
        
        else:
            #optimizer.zero_grad()
            y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            #loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            loss = custom_loss(y_pred, y_batch.to(device))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
            
            
            if i % 1000 == 0:
                print('saving model checkpoint at iteration={}'.format(i))
                torch.save(model.module.state_dict(), '{}/bert_large_fold_{}_epoch_2_ckpt_iteration_{}.bin'.format(OUTDIR, I, i))
            
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    
output_model_file = '{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE)
torch.save(model.module.state_dict(), output_model_file)
'''
######################################################################

#stage 4 begins

print('stage 4 begins')
lr, MIN, MAX, STAGE, SEED, WARMUP, FACTOR = generate_params(epoch_len, lr=lr, STAGE=4, num_stages=4)
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)
#model = nn.DataParallel(model)
model.load_state_dict(torch.load('{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE-1)))
#model = model.module
model = model.to(device)
#model = model.cuda()

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

num_train_optimization_steps = int(FACTOR*2*EPOCHS * len(train_loader.tensors[0]) /batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=lr,
                     warmup=WARMUP,
                     t_total=num_train_optimization_steps)
                      
model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)


#######################
# multi-gpu
#######################
model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5])

model=model.train()


del param_optimizer, optimizer_grouped_parameters
gc.collect()

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
tq = tqdm(range(EPOCHS))
for epoch in tq:
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    # zero.grad() moved ahead
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
        if i < MIN:
            pass
        elif i > MAX:
            pass
        
        else:
            #optimizer.zero_grad()
            y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            #loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            loss = custom_loss(y_pred, y_batch.to(device))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
            
            
            if i % 1000 == 0:
                print('saving model checkpoint at iteration={}'.format(i))
                #torch.save(model.module.state_dict(), '{}/bert_large_fold_{}_epoch_2_ckpt_iteration_{}.bin'.format(OUTDIR, I, i))
                torch.save(model.module.state_dict(), '{}/bert_large_full_epoch_2_ckpt_iteration_{}.bin'.format(OUTDIR, i))

    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)
    
#output_model_file = '{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, STAGE)
output_model_file = '{}/bert_large_full_lr_{}_STAGE_{}.bin'.format(OUTDIR, LR, STAGE)
torch.save(model.module.state_dict(), output_model_file)

######################################################################

print('starting evaluation')

bs = 64
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)
#model = nn.DataParallel(model)
model.load_state_dict(torch.load(output_model_file))
#model = model.module
model.to(device)
for param in model.parameters():
    param.requires_grad=False
model.eval()
valid_preds = np.zeros((len(X_val)))
valid = torch.utils.data.TensorDataset(torch.tensor(X_val,dtype=torch.long))
valid_loader = torch.utils.data.DataLoader(valid, batch_size=bs, shuffle=False)

tk0 = tqdm_notebook(valid_loader)
for i,(x_batch,)  in enumerate(tk0):
    pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
    valid_preds[i*bs:(i+1)*bs]=pred[:,0].detach().cpu().squeeze().numpy()


MODEL_NAME = 'model1'
test_df[MODEL_NAME]=torch.sigmoid(torch.tensor(valid_preds)).numpy()
TOXICITY_COLUMN = 'target'
bias_metrics_df = compute_bias_metrics_for_model(test_df, identity_columns, MODEL_NAME, 'target')
bias_metrics_df
print(get_final_metric(bias_metrics_df, calculate_overall_auc(test_df, MODEL_NAME)))

test_df.to_csv('{}/test_fold_{}_lr_{}.csv'.format(OUTDIR, I,lr))
bias_metrics_df.to_csv('{}/test_bias_metrics_df_{}_lr_{}.csv'.format(OUTDIR, I, lr))


######################################################################
'''
model = BertForSequenceClassification.from_pretrained("../working2", cache_dir=None, num_labels=7)

model.load_state_dict(torch.load('{}/bert_large_fold_{}_lr_{}_STAGE_{}.bin'.format(OUTDIR, I, LR, 4)))
model = model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

# reduce FACTOR ACCORDINGLY
FACTOR = 0.2
print('Starting extra epoch with FACTOR = {} and lr = {}'.format(FACTOR, extra_lr))

num_train_optimization_steps = int(FACTOR*2*EPOCHS * len(train_loader.tensors[0]) /batch_size/accumulation_steps)

optimizer = OpenAIAdam(optimizer_grouped_parameters,
                     lr=extra_lr,
                     warmup=0,
                     t_total=num_train_optimization_steps)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1",verbosity=0)

#######################
# multi-gpu
#######################
model=nn.DataParallel(model, device_ids=[0,1,2,3,4,5])

model=model.train()
del param_optimizer, optimizer_grouped_parameters
gc.collect()

print('remaining cuda memory is: {}'.format(torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device=None) ))
np.random.seed(SEED*4)
torch.manual_seed(SEED*4)
torch.cuda.manual_seed(SEED*4)
torch.backends.cudnn.deterministic = True
tq = tqdm(range(EPOCHS))


MIN = 0
MAX = 7500

for epoch in tq:
    avg_loss = 0.
    avg_accuracy = 0.
    lossf=None
    tk0 = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
    # zero.grad() moved ahead
    optimizer.zero_grad()
    for i,(x_batch, y_batch) in tk0:
        if i < MIN:
            pass
        elif i > MAX:
            pass
        else:
            #optimizer.zero_grad()
            y_pred = model(x_batch.to(device), attention_mask=(x_batch>0).to(device), labels=None)
            #loss =  F.binary_cross_entropy_with_logits(y_pred,y_batch.to(device))
            loss = custom_loss(y_pred, y_batch.to(device))
            
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            if (i+1) % accumulation_steps == 0:             # Wait for several backward steps
                optimizer.step()                            # Now we can do an optimizer step
                optimizer.zero_grad()
            if lossf:
                lossf = 0.98*lossf+0.02*loss.item()
            else:
                lossf = loss.item()
            tk0.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(((torch.sigmoid(y_pred[:,0])>0.5) == (y_batch[:,0]>0.5).to(device)).to(torch.float) ).item()/len(train_loader)
            
            if i % 500 == 0:
                print('saving model checkpoint at iteration={}'.format(i))
                torch.save(model.module.state_dict(), '{}/bert_large_fold_{}_epoch_3_ckpt_iteration_{}.bin'.format(OUTDIR, I, i))
    tq.set_postfix(avg_loss=avg_loss,avg_accuracy=avg_accuracy)

 '''   
