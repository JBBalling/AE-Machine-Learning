# Performance metrics


$Accuracy = \frac{TP+TN}{TP+TN+TP+FP}$
does not penalize false positives and false negatives!


$Sensitivity = \frac{TP}{TP+FN}$

$Specificity = \frac{TN}{TN+FP}$

$Precision = \frac{TP}{TP+FP}$

$F1 score = \frac{2*sensitivity*precision}{sensitivity+precision}$

Higher sensitivity &rarr; less FN!
Higher specifity &rarr; less FP!


Decision Point - threshold on which to decide on either class.

ROC Receiver Operating Curve
&rarr; to select 

AUC - Area under the Curve &rarr; to automatically compare models.


## Validation

- holdout validiation
- k-fold cross-validation: change training of model with different folds of the input data set -> changing training, validation and test set through iterations
- leave-one-out