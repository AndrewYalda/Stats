# Loading Packages

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Loading Data
s = pd.read_csv("social_media_usage.csv")


# Creating function
def clean_sm(x):
    return np.where(x == 1,1,0)

# Creating Test DataFrame
toy = pd.DataFrame(np.arange(6).reshape(-1, 2))


# Subsetting dataframe
ss = s[['web1h','age', 'gender', 'income','educ2','par','marital']]
# Using function to create sm_li column
sm_li = clean_sm(ss.web1h)
# Drop web1h column
ss = ss.drop(columns=['web1h'])
# Adding sm_li column
ss.insert(loc=0, column='sm_li', value=sm_li)

# Dropping Don't know or refused for income
ss = ss[ss.income.isin([98,99]) == False]
# Droping Don't Know for Age
ss = ss[ss.age.isin([99]) == False]
# Dropping Don't know or refused for Gender
ss = ss[ss.gender.isin([98,99]) == False]
# Dropping Don't know or refused for marital status
ss = ss[ss.marital.isin([8,9]) ==False]
# Dropping Don't Know or Refused for par
ss = ss[ss.par.isin([8,9]) == False]
# Dropping Don't Know or Refused for educ2
ss = ss[ss.educ2.isin([98,99]) == False]

y = ss["sm_li"]
X = ss[['age', 'gender', 'income', 'educ2', 'par','marital']]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,   
                                                    random_state=123)


lr = LogisticRegression(class_weight="balanced")

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

ss_conf_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Real negative","Real positive"])


# Precision: TP / (TP+FP)
precision = ss_conf_matrix.iloc[1,1] / (ss_conf_matrix.iloc[1,1] + ss_conf_matrix.iloc[1,0])

# Recall: TP / (TP+FN)
recall = ss_conf_matrix.iloc[1,1] / (ss_conf_matrix.iloc[1,1] + ss_conf_matrix.iloc[0,1])

# F1 score 2 * (Precision * Recall)/(Precision + Recall)
score_f1 = 2 * ((precision * recall) / (precision + recall))


person_1 = pd.DataFrame([[42,1,8,7,0,1]],
                        columns=["age", "gender", "income", "educ2", "par", "marital"])
person_2 =  pd.DataFrame([[82,1,8,7,0,1]],
                        columns=["age", "gender", "income", "educ2", "par", "marital"])


predicted_class_1 = lr.predict(person_1)
probs_1 = lr.predict_proba(person_1)


predicted_class_2 = lr.predict(person_2)
probs_2 = lr.predict_proba(person_2)
