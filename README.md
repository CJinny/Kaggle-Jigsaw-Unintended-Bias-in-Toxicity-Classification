# Kaggle-Jigsaw-Unintended-Bias-in-Toxicity-Classification
NLP competition on Kaggle

## Summary of this competition
 - **Summary**: This is a Natural Language Processing competition in which we were asked to detect toxic comments and at the same time reduce prediction errors specifically for comments that mention certain identities (unintended bias).
 - **Challenge**: Our goal was not just to maximize area under curve (AUC) of our model with regards to one target, but also minimize unintended bias with comments which mention a group of 9 identities. The evaluation metric uses a unique Jigsaw bias AUC which takes consideration of both the overall AUC and Bias AUC. Please refer to this [link](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) for more details about this metric.

## Our Strategies

- **Add weights and use additional identity labels in our loss function**: Our loss function is as follows:

 ```
 weights = np.ones(len(train_df)) / 4
 weights += (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
 weights += (( (train_df['target'].values >= 0.5).astype(bool).astype(np.int) +
    (train_df[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
 weights += (( (train_df['target'].values < 0.5).astype(bool).astype(np.int) +
    (train_df[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
 ```

 ```
 def custom_loss(y_pred, y_true):
     bce_loss_1 = F.binary_cross_entropy_with_logits(y_pred[:,:1].reshape(-1), y_true[:,:1].reshape(-1), weight=y_true[:,1:2].reshape(-1))   
     N = y_true.shape[1]
     custom_loss = bce_loss_1
     for i in range(N-2):
         custom_loss += F.binary_cross_entropy_with_logits(y_pred[:,1+i], y_true[:,2+i])
     return custom_loss
 ```
Basically, every comment gets an initial weight of 0.25, then comments which mention each identity (**Subgroup**) will get a 0.25 additional weight, then comments which are toxic but didn't mention identity will get 0.25 additional weight, and last comments which are not toxic but mention identity will get 0.25 additional weight. The custom loss will be considering both overall target prediction loss as well as loss of auxilary labels.

- **Combine a variety of models**. We started with a bilateral LSTM model using FastAI framework, but then quickly move to a more powerful model: BERT(Bidirectional Encoder Representations from Transformers). The following pretrained models were used:

  - BERT-Large, Uncased (Whole Word Masking) 
  - BERT-Base, Uncased
  - GPT-2, Uncased
  
  We have discovered that `OpenAIAdam` performs better than `BertAdam` on validation set, hence we chose to use `OpenAIAdam` for our model optimizer.

- **We also attempted many model frameworks**:
    - Pytorch, Many thanks to [hunggingface](https://github.com/huggingface/pytorch-pretrained-BERT) for this wonderful Pytorch model for BERT. The cool thing with pytorch BERT is that it enables us to use `gradient_accumulation_steps` to control how frequently we want gradients to be accumulated.
    - FastAI, The cyclical learning rates are always useful.
    - Keras, I haven't used Keras but my teammate had and was able to achieve good leaderboard score with it.

  For my pytorch BERT base model training use a 4-stage approach to train our models, namely: 
    1. Stage 1: `lr=2e-5`, `num_train_optimization_steps = int(2*len(train_loader)/batch_size/accumulation_steps)`
    2. Stage 2: `lr=1.5e-5`, `num_train_optimization_steps = int(1.5*2*len(train_loader)/batch_size/accumulation_steps)`
    3. Stage 3: `lr=1e-5`, `num_train_optimization_steps = int(1*2*len(train_loader)/batch_size/accumulation_steps)`
    4. Stage 4: `lr=0.5e-5`, `num_train_optimization_steps = int(0.5*2*len(train_loader)/batch_size/accumulation_steps)`

  Each stage will train 0.5 epoch so 2 epochs in total. In order to make our model train train faster for BERT-Large, we also increased our batch-size from 32 to 96.


What I have learned from this competition:


What we could have done better:

- **Use a better custom loss function**. We attempted to incorporate identity columns in our custom loss functioon but were not successful during model training.

- **Use static features**. We could have added additional features such as number of exclamation marks or number of emojis which could have been useful.

- **Use the XLNet pretrained model**. This mdoel was published just a few days before the competition ends, but we could have done better.






