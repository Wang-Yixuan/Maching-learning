import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model._logistic import LogisticRegression

#数据处理
file = open('dataR2.txt','r')
data = []
for line in file.readlines():
    row = []
    lines = line.strip().split('\t')
    for x in lines:
        row.append(x)
    data.append(row)
del data[0]
dataframe = pd.DataFrame(data, columns=['Age','BMI','Glucose','Resistin','Classification'])
X = dataframe[['Age','BMI','Glucose','Resistin']].values
#dataframe = pd.DataFrame(data, columns=['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1','Classification'])
#X = dataframe[['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1']].values
Y = dataframe['Classification'].values

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

#决策树
clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3)
clf = clf.fit(Xtrain, Ytrain)
feature_name = ['Age', 'BMI', 'Glucose', 'Resistin']
target_name = ['Yes', 'No']
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_name, class_names=target_name, filled=True, rounded=True, special_characters=True)
plot = pydotplus.graph_from_dot_data(dot_data)
plot.write_pdf('DT.pdf')
y_pred = clf.predict(Xtest)
y_score = clf.predict_proba(Xtest)[:, 0]

#贝叶斯分类器
gnb = GaussianNB()
gnb = gnb.fit(Xtrain, Ytrain)
y_pred = gnb.predict(Xtest)
y_score = gnb.predict_proba(Xtest)[:, 0]

#逻辑回归
clf = LogisticRegression()
clf.fit(Xtrain, Ytrain)
y_pred = clf.predict(Xtest)
y_score = clf.predict_proba(Xtest)[:, 0]

#ROC曲线与AUC值
fpr, tpr, thersholds = roc_curve(Ytest, y_score, pos_label='1',drop_intermediate=False)
AUC = auc(fpr, tpr)
print(AUC)
plt.plot(fpr, tpr, color='black', lw=2, label='ROC curve (area = %0.2f)' % AUC)
plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()



