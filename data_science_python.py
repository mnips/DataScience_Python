import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas
print df.head()
# print df.describe() #Gives summary of dataset
#sns.boxplot(data=df['ApplicantIncome'])
df.boxplot(column='ApplicantIncome',by="Education")
#plt.show()

class0=class1=float(0)
freq0=freq1=float(0)
for index,row in df.iterrows():
    if row['Credit_History']==0:
        class0=class0+int(row['Loan_Status']=='Y')
        freq0=freq0+1
    elif row['Credit_History']==1:
        class1=class1+int(row['Loan_Status']=='Y')
        freq1=freq1+1

print 'Probility of getting loan for each Credit History class:'
print 'Class 0:',class0/freq0
print 'Class 1:',class1/freq1
# ADDITION OF PLOT OF ABOVE SHORT ANALYSIS
fig = plt.figure(figsize=(8,4))

ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applicants')
ax1.set_title("Applicants by Credit_History")

ax1.bar(np.arange(2),[freq0,freq1],width=0.5)
ax1.set_xticks(np.arange(2))
ax2 = fig.add_subplot(122)
ax2.bar(np.arange(2),[class0/freq0,class1/freq1],width=0.5)
ax2.set_xticks(np.arange(2))
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probability of getting loan')
ax2.set_title("Probability of getting loan by credit history")
plt.show()
plt.close()
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
plt.show()
plt.close()
###CHECKING NUMBER OF MISSING VALUES
df.apply(lambda x: sum(x.isnull()),axis=0)
#Imputation as high probability of not self employed
df['Self_Employed'].fillna('No', inplace=True)
#Pivot Table
table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)
# Define function to return value of this pivot_table
def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)
#Treating Extreme Values
df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20)
plt.show()