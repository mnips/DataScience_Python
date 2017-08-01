import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
    print model
    model.fit(data[predictors],data[outcome])
    # Make predictions on training set:
    predictions = model.predict(data[predictors])

    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print "Accuracy : %s" % "{0:.3%}".format(accuracy)

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=5)
    error = []
    for train, test in kf:
        # Filter training data
        train_predictors = (data[predictors].iloc[train, :])

        # The target we're using to train the algorithm.
        train_target = data[outcome].iloc[train]

        # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)

        # Record error from each cross-validation run
        error.append(model.score(data[predictors].iloc[test, :], data[outcome].iloc[test]))

    print "Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error))

    # Fit the model again so that it can be refered outside the function:
    model.fit(data[predictors], data[outcome])

df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas
print df.head()

df_test=pd.read_csv("test.csv")
#print df.isnull().any()
df = df.fillna(method='ffill')
#df['LoanAmount_log'] = np.log(df['LoanAmount'])
##SKLEARN REQUIRES DATA IN NUMERIC FORM

var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])

###LOGISTIC REGRESSION
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)
print
###DECISION TREE
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
predictor_var = ['Credit_History','Loan_Amount_Term']
classification_model(model, df,predictor_var,outcome_var)
print
###RANDOM FOREST
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'   ]
classification_model(model, df,predictor_var,outcome_var)
