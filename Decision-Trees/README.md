# Decision Trees

```
            Finding Nemo
                O
          yes /   \ no
             O     O
           DK?     DK?
       yes /  \n y/  \ no    
          O    O O    O
```

Based on a survey predict which user is going to the new movie, depending on a small sample of reviews.


## Drilling Oil

CART = Classification and Regression Trees

Where to drill for highest profit?
Per past drill site per depth: 1000m, 2000m, 3000m, ...

Features:
- porosity
- gamma ray
- density
- sonic


Labels - binary indicators per drill site:
1 &rarr; project was profitable
0 &rarr; project was not profitable


Which feature to split on and on which value should it be splitted?
Evaluate each feature:
1. evaluate each values within the feature to split on?
    1. ORDER on feature
    1. find averages between features, as split points 
        E.g (ordered on feature [0]):
        $\vec{x_1} = [0.02, 0.12, 0.83, 2300, ...]$
        $\vec{x_3} = [0.07, 0.14, 0.42, 1500, ...]$
        $\vec{x_2} = [0.13, 0.08, 0.63, 1700, ...]$
    1. average between $\vec{x_1}, \vec{x_3} = 0,045$
    1. average between $\vec{x_3}, \vec{x_2} = 0,1$
    1. split point to the next x
    1. Gini impurity $1-(P_0²-P_1²) * \frac{numfeatures in node}{total features}$ in each leaf:
    1. weight each node by their fraction
    1. first split would be on porosity on 0.1
    1. its possible to split on a feautre multiple times
    1. stop when a criteria is reached: max_depth, min samples per node, go until nodes are pure (overfitting)

$\vec{x_1}, \vec{x_2} = 1; \vec{x_3}, \vec{x_4} = 0; $
```
    {x1, x2, x3, x4}                {x1, x2, x3, x4}             {x1, x2, x3, x4}
         </ 0.045 \>=                    </ 0.1 \>=                 </ 0.17 \>=
      {x1}     {x2, x3, x4}           {x1,x2}  {x3, x4}         {x1,x2,x3}  {x4} 
  G=1-(1²-0¹)   1-(0.33²-0.66²)          
  0 * 0,25 + 0.45*0.75 = 0.3375        0 (perfect separation)       0.3375

```
How to handle missing data?
if the feature which is split on is missing in the data, go to the next best feature to split on

if another categorical variable is added, add another gini impurity for this class

# Regression Trees

1. ORDER on feature 
1. split point to the next x
1. $Mean Squared Error (MSE) = \frac{\sum_i^N{(x_i - \overline{x})²}}{n}$
1. sum the error for each split and divide through the number of samples
1. take the split with the lowest mse from the step above
1. predict: average of the values in the node will be assigned


with Binary Class:

Features:
- porosity
- gamma ray
- density
- sonic
- northern / southern hemisphere

or categorical class:

Features:
- porosity
- gamma ray
- density
- sonic
- country



Boosting:
train another tree on the error of the first tree
- ensamble of weak learners

Bossted-Trees:
- easily overfit
    - cross-validate
    - start with 100 or depth = 3

Bagging:
- sampling technique
- sample the dataset with replacement and train a tree with each sample.
- reduces variance



Random Forests
Average(Tree, Tree, Tree)

C4.5:
- desgined for classification
- n-ary splits
- information gain (entropy)


Libraries:
XGBoost
LightGBM
CatBoost
