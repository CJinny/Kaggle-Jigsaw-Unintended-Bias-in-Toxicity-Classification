# Kaggle-Jigsaw-Unintended-Bias-in-Toxicity-Classification
Jigsaw Unintended Bias in Toxicity Classification competition on Kaggle

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
Basically, every comment gets an initial weight of 0.25, then comments which mention each identity (**Subgroup**) will get a 0.25 additional weight, then comments which are toxic but didn't mention identity will get 0.25 additional weight, and last comments which are not toxic but mention identity will get 0.25 additional weight. The custom loss will be considering both overall target prediction loss as well as loss of identity predictions.

- **Combine a variety of models**:





