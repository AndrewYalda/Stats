import streamlit as st
import pandas as pd

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





# Linkedin Prediction Function



st.title('Linkedin Prediction Function')
st.subheader('Andrew Yaldaparast')
st.header('This application is used to predict if someone is a Linkedin user based on demographic inputs ')





educ = st.selectbox("Specify Education Level",
    options = ["Less than high school (Grades 1-8 or no formal schooling)",
        "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)",
        "High school graduate (Grade 12 with diploma or GED certificate",
        "Some college, no degree (includes some community college)",
        "Two-year associate degree from a college or university",
        "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)",
        "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)",
        "Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD"])


if educ == "Less than high school (Grades 1-8 or no formal schooling)":
    educ = 1
elif educ == "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)":
    educ = 2
elif educ ==  "High school graduate (Grade 12 with diploma or GED certificate":
    educ = 3
elif educ ==  "Some college, no degree (includes some community college)":
    educ = 4
elif educ == "Two-year associate degree from a college or university":
    educ = 5
elif educ == "Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)":
    educ = 6
elif educ == "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)":
    educ = 7
else:  educ = 8



income = st.slider(label="Specify Income (Annual)",
    min_value=1,
    max_value=250000,
    value=50000
)

if income >150000:
    income = 9
elif income > 100000:
    income = 8
elif income > 75000:
    income = 7
elif income > 50000:
    income = 6
elif income > 40000:
    income = 5
elif income >30000:
    income = 4
elif income > 20000:
    income = 3
elif income > 10000:
    income = 2
else: income = 1

par = st.selectbox("Are you a parent of a child under 18 living in your home?",
    options = ["Yes",
    "No"
    ]
)

if par == "Yes":
    par = 1
else: par = 2


marital = st.selectbox("Specify current marital status",
    options = ["Married",
    "Living with a partner",
    "Divorced",
    "Separated",
    "Widowed",
    "Never been married"
    ]
)

if marital == "Married":
    marital = 1
elif marital == "Living with a partner":
    marital = 2
elif marital ==  "Divorced":
    marital = 3
elif marital ==  "Separated":
    marital = 4
elif marital == "Widowed":
    marital = 5
else: marital = 6


gender = st.selectbox("Specify gender",
    options = ["Male",
    "Female",
    "other",
    ]
)

if gender == "Male":
    gender = 1
elif gender == "Female":
    gender = 2
else: gender = 3

age = st.slider(label="Specify age",
    min_value=18,
    max_value=97,
    value=28
)

prediction =  pd.DataFrame([[age,gender,income,educ,par,marital]],
                        columns=["age", "gender", "income", "educ2", "par", "marital"])



prediction_predicted_class = lr.predict(prediction)
prediction_probs = lr.predict_proba(prediction)
result = round(prediction_probs[0,1],3)*100


st.metric('Percent probability this person is a Linkedin user:', result)



