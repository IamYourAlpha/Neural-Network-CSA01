#############################################################################
#
# Author      :  Chowdhury Md Intisar
# Institution :  University of Aizu 
# Language    :  Python ( 2.7 )
# Problem     :  Predicting Titanic Problem Survival
# Source      :  Kaggle 
#
#
#
##########################################################################


import pandas as pd
import seaborn as sns
import numpy as np
import math
import matplotlib.pyplot as plt

# import all the necessary data for training and testing
                                                      #
df_train = pd.read_csv('/home/in2/Documents/abc_python/Kaggle/Titanic/train.csv')                   #  
df_test = pd.read_csv('/home/in2/Documents/abc_python/Kaggle/Titanic/test.csv')                     #
df_combine = [df_train, df_test]                      #
                                                      #
#######################################################

# Checking the data details and types 
print df_train.info()
print 40*'='
print df_test.info()
######################################################


print df_train[['Survived', 'Pclass']].groupby('Pclass').mean()

for i in df_combine:
    i['Sex'] = i['Sex'].map( { 'female' : 0, 'male' : 1 } )
print df_train[['Sex', 'Survived']].groupby('Sex').mean()

for df in df_combine:
    df['Title'] = df.Name.str.extract( '([A-Za-z]+)\.', expand = False)
print pd.crosstab( df_train['Title'], df_train['Sex'] )
print 50*'='
print pd.crosstab( df_train['Title'], [ df_train['Sex'], df_train['Survived'] ] ) 
print 50*'='
print pd.crosstab( df_test['Title'], df_test['Sex'] ) 
print 50*'='
for df  in df_combine:
    df['Title'] = df['Title'].replace( ['Capt','Don','Dona','Jonkheer','Rev'], 'Rare')
    df['Title'] = df['Title'].replace( ['Lady','Mlle', 'Mme','Ms'],            'Miss')
    df['Title'] = df['Title'].replace( ['Countess'],                           'Mrs')
    df['Title'] = df['Title'].replace( ['Sir'],                                'Mr')
    df['Title'] = df['Title'].replace( ['Col','Dr','Major'],                  'Profession')

print pd.crosstab( df_train['Title'], [ df_train['Sex'], df_train['Survived'] ] )
#print df_train['Title'] 

print df_train[['Title', 'Survived']].groupby('Title').mean() 

for df in df_combine:
    df['Title'] = df['Title'].map( { 'Mrs' : 0 , 'Miss' : 1, 'Master' : 2, 'Profession' : 3, 'Mr' : 4, 'Rare' : 5 } ) 

#print df_train['Title'] 

age_guess  = np.zeros( (2,3,5) ) # Sex-2, PClass-3, Title-5 

for df in df_combine:
    df_noAge  = df[ df['Age'].isnull() ]
    for sex in np.arange(0,2):
        for pclass in np.arange(1,4):
            for title in np.arange(0,5):
                if len( df_noAge[ (df_noAge.Sex == sex) & (df_noAge.Pclass == pclass) &
                    (df_noAge.Title == title) ] ):
                      age_guess[sex, pclass-1, title] = int( math.floor( df[ (df.Sex == sex) &
                          (df.Pclass == pclass) & (df.Title == title) ]['Age'].dropna().mean() ) ) + 0.5
                      df.loc[ df.Age.isnull() & ( df.Sex == sex ) & ( df.Pclass == pclass ) &
                              (df.Title == title), 'Age'] = age_guess[sex, pclass-1, title]
                      print 'The guess age is :', age_guess[sex, pclass-1, title]


print df_train[ [ 'Age', 'Survived'] ].groupby('Age').mean()

for df in df_combine:
    df['isInfant'] = 0
    df['isKid'] = 0
    df['isOld'] = 0
    df.loc[ (df['Age'] < 1), 'isInfant' ] = 1
    df.loc[ (df['Age'] >=1) & (df['Age']<6), 'isKid'] = 1
    df.loc[ (df['Age']>=64), 'isOld'] = 1
print 50*'='
print df_train[['isInfant', 'Survived']].groupby('isInfant').mean()
print df_train[['isKid', 'Survived']].groupby('isKid').mean()
print df_train[['isOld', 'Survived']].groupby('isOld').mean()
print 50*'='

for df in df_combine:
    df['band'] = pd.qcut(df['Age'], 4)

print df_train[['band', 'Survived']].groupby('band').mean()
print 50*'='
df_train.drop(labels='band', inplace=True, axis = 1)


for df in df_combine:
    df['AgeBand'] = 0;
    df.loc[ (df['Age'] < 21) ,                        'AgeBand'] = 0
    df.loc[ (df['Age'] <= 28.5) & (df['Age'] >21),    'AgeBand'] = 1
    df.loc[ (df['Age'] <= 36.75) & (df['Age'] > 28.5),'AgeBand'] = 2
    df.loc[ (df['Age'] >36.75) ,                      'AgeBand'] = 3
print df_train['AgeBand']
print 50*'='
print df_train[['SibSp', 'Survived']].groupby('SibSp').mean()
print 50*'='
print df_train[['Parch', 'Survived']].groupby('Parch').mean()
 

for df in df_combine:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
print df_train[['FamilySize', 'Survived']].groupby('FamilySize').mean()


for df in df_combine:
    df['isAlone'] = 0
    df.loc[ df['FamilySize'] ==  1, 'isAlone'] = 1
print df_train[[ 'isAlone', 'Survived']].groupby('isAlone').mean()


for df in df_combine:
    df['isLargeFamily'] = 0
    df.loc[ df['FamilySize'] > 4, 'isLargeFamily'] = 1

print df_train[[ 'isLargeFamily', 'Survived']].groupby('isLargeFamily').mean()

print 50*'='

for df in df_combine:
    df['isSpecialTicket'] = df.Ticket.str.extract( '([A-Z])', expand=False)
    df['isSpecialTicket'] = df['isSpecialTicket'].fillna('RE')

print pd.crosstab( df_train['isSpecialTicket'], df_train['Survived'] )

print df_train[ ['isSpecialTicket', 'Survived'] ].groupby('isSpecialTicket').mean()

for df in df_combine:
    df['isSpecialTicket'] = df['isSpecialTicket'].map( { 'P':0, 'F':1, 'RE':2, 'C':3, 'S':4, 'L':5, 'W':6, 'A':7})

print df_train[ ['Survived', 'isSpecialTicket']].groupby('isSpecialTicket').mean()

print df_test[ df_test.Fare.isnull()]

df_test.loc[ df_test.Fare.isnull(), 'Fare' ]  = df_test[ df_test.Pclass == 3 ]['Fare'].median()


grid = sns.FacetGrid( df_train, col = 'Survived')
grid.map(plt.hist, 'Fare')
#plt.show()

df_train[ 'FareBandTmp' ] = pd.qcut( df_train['Fare'], 4)
print df_train[ ['FareBandTmp', 'Survived']].groupby('FareBandTmp').mean()

df_train.drop( labels = 'FareBandTmp', axis = 1, inplace = True)

for df in df_combine:
    df['FareBand'] = 0
    df.loc[ ( df.Fare <= 7.91 ),                        'FareBand'] = 0
    df.loc[ ( df.Fare <= 14.454 ) & (df.Fare> 7.91) ,   'FareBand'] = 1
    df.loc[ ( df.Fare <= 31.0  ) & ( df.Fare >14.454),  'FareBand'] = 2
    df.loc[ ( df.Fare > 31.0 ),                         'FareBand'] = 3
    
print pd.crosstab( df_train['FareBand'], df_train['Survived'] )
#print df_train['FareBand']

for df in df_combine:
    df['Cabin'] = df['Cabin'].str.extract( '([A-Z])', expand=False)
    df['Cabin'] = df['Cabin'].fillna('Z')
print df_train[ [ 'Cabin', 'Survived' ] ].groupby('Cabin').mean()

for df in df_combine:
    df['Cabin'] = df['Cabin'].map( { 'D':0, 'E':1, 'B':2, 'F':3, 'C':4, 'G':5, 'A':6, 'Z':7, 'T':8 })


print pd.crosstab( df_train.Survived, df_train.Embarked)
print pd.crosstab( df_train.FareBand, df_train.Embarked)


for df in df_combine:
    df.loc[ df['Embarked'].isnull(), 'Embarked'] = 'S'

print df_train[ [ 'Survived', 'Embarked'] ].groupby('Embarked').mean()

for df in df_combine:
    df['Embarked'] = df['Embarked'].map( { 'C':0, 'Q':1, 'S':2 } )
print df_train[ [ 'Survived', 'Embarked'] ].groupby('Embarked').mean()


#print df_train.dtypes.index.values 

print df_train.dtypes.index.values

########################################################
#
#
#  Modeling and predicting with Learning Algorithm 
#
#
#
##########################################################






selected_features = [  'Pclass',  'Sex',   'Cabin', 'Embarked', 'Title', 'isInfant', 'isKid', 'isOld',
  'AgeBand', 'FamilySize', 'isAlone', 'isLargeFamily', 'isSpecialTicket',
   'FareBand']
x_train = df_train[ selected_features ] 
y_train = df_train[ 'Survived' ] 
x_test = df_test[ selected_features ] 



##############################################################
#
#
# LOADING THE MACHINE LEARNING LIBRARY 
#
#
##############################################################


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression 


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection  import GridSearchCV


clf = LogisticRegression( C = 5)
clf.fit(x_train, y_train)

df_coeff = pd.DataFrame( x_train.columns.delete(0) )
#df_coeff.columns = [ 'Feature']
df_coeff['C']  = pd.Series( clf.coef_[0] ) 
df_coeff.columns =  [ 'Feature',  'Correlation' ] 
print df_coeff.sort_values( by= 'Correlation', ascending=False)

clfs = [ SVC(probability = True), RandomForestClassifier(n_estimators=1000),
        AdaBoostClassifier(n_estimators=1000),
        GradientBoostingClassifier(n_estimators=1000),
        KNeighborsClassifier(n_neighbors=5),
        LogisticRegression(),
       ]
score_acc = []
score_f1  = []

'''
for i in np.arange(0, len(clfs)):
    clf = clfs[i]
    clf.fit( x_train, y_train )
    y_pred = clf.predict(x_train)
    score_acc.append( accuracy_score(y_pred, y_train) )
    score_f1.append( f1_score( y_pred, y_train ) )

cols  = [ 'SVM', 'RandomForest', 'Adaboost', 'GradientBoosting', 'KNeigbors', 'LogisticReg']
scores = pd.DataFrame( data = [ score_acc, score_f1],  columns  =  cols, index = [ 'accuracy', 'f1'])
print scores
'''
###################################################
#
#
# We will choose Random Forest because it has high accuracy 
#
#
#################################################

clf_chosen = RandomForestClassifier(n_estimators=1000)
clf_chosen.fit(x_train, y_train)
y_pred = clf_chosen.predict( x_train )
y_pred_test = clf_chosen.predict( x_test )

result = pd.DataFrame( { 'PassengerId' : df_test['PassengerId'], 'Survived':
                        y_pred_test
                       })
print result 






