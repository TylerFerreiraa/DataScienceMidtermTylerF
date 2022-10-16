# Data Analysis

This is the flow of the analysis I have performed with these specific Dataset. Now I will explore more in details every step.

<p align="center">
  <img src="https://i.ibb.co/wMnvvRK/tyler-flow.png" />
</p>

## Load Data
Imported TSV data into my environment. Data has been previously cleaned from unnecessary rows.

## Visual Exploration

I have plotted the most important statistics for each numerical column (min, max, mean, standard deviation, range) and also inspected the target variable. This step allowed me to understand that the dataset is highly imbalanced and the ratio of Class-1 to Class-2 instances is 250:50, more concisely 5:1. Some Oversampling techniques will be applied in future steps.

## Data preprocessing
In this step, I have performed the following actions:
- I explored the correlation matrix to understand the correlation among columns. Then, I removed columns highly corrrelates since they won't impact the model's performances
- Scaled (via MinMaxScaler) the numerical columns in my dataset
- Encoded my categorical variables via LabelEncoder. If you are not familiar with Encoding, it basically means that I assigned a number to each of the categorical values in the dataset

## Model's Parameters Grid Search
This is the step where I spent the most time on. The first decision I had to take was about which models to study. I opted for:
- LogisticRegression
- RandomForestClassifier
- AdaBoostClassifier
- Support Vector Machines
- XGBoost
- Neural Networks
- DecisionTreeClassifier

I have used the famous library sklearn to implemente all the given models. For every model evaluation I have split the dataset into training and test with a percentage of 80% and 20%. Given the small numerosity of the dataset I have made the split for every iteration, that way our results will be more robust.

I will now give an overview of all the models. 

### LogisticRegression

<p align="center">
  <img src="https://user-images.githubusercontent.com/44615027/195816767-63ecde0f-b773-490f-9acb-55646b955241.png" />
</p>
<p>
In statistics, the logistic model (or logit model) is a statistical model that models the probability of an event taking place by having the log-odds for the event be a linear combination of one or more independent variables. In regression analysis, logistic regression[1] (or logit regression) is estimating the parameters of a logistic model (the coefficients in the linear combination). Formally, in binary logistic regression there is a single binary dependent variable, coded by an indicator variable, where the two values are labeled "0" and "1", while the independent variables can each be a binary variable (two classes, coded by an indicator variable) or a continuous variable (any real value). In Our scenario, the categorical variables have been encoded.</p>

Logistic Regression had the following results: **Accuracy: 0.84 Precision Score 0.91 Recall 0.88 F1 Score 0.89**    
Best parameters configurations: **{'C': 0.001, 'penalty': 'l2', 'solver': 'newton-cg'}**


### RandomForestClassifier
<p align="center">
  <img src="https://miro.medium.com/max/1200/1*hmtbIgxoflflJqMJ_UHwXw.jpeg" />
</p>
<p>
The Random forest classifier creates a set of decision trees from a randomly selected subset of the training set. It is basically a set of decision trees (DT) from a randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction.</p>
    

RandomForestClassifier had the following results: **Accuracy: 0.83 Precision Score 0.81 Recall 1.0 F1 Score 0.9**    
Best parameters configurations: **{'bootstrap': True, 'max_depth': 20, 'n_estimators': 200}**

### AdaBoostClassifier

<p>
AdaBoost, short for Adaptive Boosting, is a statistical classification meta-algorithm formulated by Yoav Freund and Robert Schapire in 1995, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance. The output of the other learning algorithms ('weak learners') is combined into a weighted sum that represents the final output of the boosted classifier. Usually, AdaBoost is presented for binary classification, although it can be generalized to multiple classes or bounded intervals on the real line. Check the docs: https://towardsdatascience.com/understanding-adaboost-2f94f22d5bfe</p>


AdaBoostClassifier had the following results: **Accuracy: 0.84 Precision Score 0.88 Recall 0.92 F1 Score 0.9**    
Best parameters configurations: **{'n_estimators': 50}**


### Support Vector Machines

<p align="center">
  <img src="https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png" />
</p>
<p>
The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points. To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.

</p>  

SVM had the following results: **Accuracy: 0.83 Precision Score 0.92 Recall 0.85 F1 Score 0.88**    
Best parameters configurations: **{'C': 0.1, 'gamma': 1, 'kernel': 'linear'}**


### XGBoost
<p>
 it is a gradient boosting algorithm that uses decision trees as its “weak” predictors. Beyond that, its implementation was specifically engineered for optimal performance and speed.

Historically, XGBoost has performed quite well for structured, tabular data. If you are dealing with non-structured data such as images, neural networks are usually a better option.</p>
    

XGBoost had the following results: **Accuracy: 0.81 Precision Score 0.84 Recall 0.92 F1 Score 0.88**    
Best parameters configurations: **{'booster': 'gbtree', 'colsample_bytree': 0.6, 'gamma': 0.5, 'max_depth': 3, 'min_child_weight': 1, 'subsample': 0.6}**


### Neural Networks
<p align="center">
  <img height="400" src="https://machinelearningmastery.com/wp-content/uploads/2021/08/neural_networks_21.png" />
</p>
<p>
A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. In this sense, neural networks refer to systems of neurons, either organic or artificial in nature.</p>
    
Neural Networks had the following results: **Accuracy: 0.67 Precision Score 0.78 Recall 0.77 F1 Score 0.78**    
Best parameters configurations: **{'activation': 'sigmoid', 'learning_rate': 'constant', 'solver': 'lbfgs'}**


### DecisionTreeClassifier

<p align="center">
  <img height="400" src="https://venngage-wordpress.s3.amazonaws.com/uploads/2019/08/what-is-a-decision-tree-5.png" />
</p>
<p>
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.</p>
    

DecisionTreeClassifier had the following results: **Accuracy: 0.79 Precision Score 0.83 Recall 0.9 F1 Score 0.86**    
Best parameters configurations: **{'criterion': 'gini', 'max_depth': 1, 'max_features': 'auto', 'splitter': 'best'}**


## Conclusions

The final results of my analysis is that most promising model seems to be ADABoost with a F1 Score of 0.9 (Accuracy: 0.84 Precision Score 0.88 Recall 0.92).

| Metric    |   Score   |
|---    |---    |
| F1 Score      |   0.9 |
| Accuracy      |   0.84 |
| Precision     |   0. 88 |
| Recall    |       0.92 |

In the notebook, you can explore my results and have a quick understanding about which were the metrics used to assess the results. Given the nature of the dataset (highly unbalanced), my focus was on the F1 score.  To conclude the analysis, this are the variables that contributed the most towards the end results:




<a ><img src="https://i.ibb.co/mNvTbTs/index.png" alt="index" border="0"></a>
