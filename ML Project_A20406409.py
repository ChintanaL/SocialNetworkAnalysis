
# coding: utf-8

# In[108]:


#get_ipython().run_line_magic('matplotlib', 'notebook')
import numpy as np
import pandas as pd
from __future__ import division
cleanCrime = pd.read_csv('./Crime Prediction Data/Crime Prediction Data/communities-crime-clean.csv')
np.random.seed(999)

#print(cleanCrime.head())
#print (cleanCrime.dtypes)


# In[109]:


# Decision Trees
#a.	Create a new field “highCrime” which is true if the crime rate per capita (ViolentCrimesPerPop) is greater than 0.1
cleanCrime['highCrime']=(cleanCrime['ViolentCrimesPerPop'] > 0.1).astype(int)
cleanCrime.head()


# In[110]:


# Percentage of positive and negative instances
#Count
print("Positive Intances count",sum(cleanCrime['highCrime']==1))
print("Negative Instances count",sum(cleanCrime['highCrime']==0))
print("Total Count",cleanCrime.highCrime.count())
#Percentage
print("Positive Instances Percentage",(sum(cleanCrime['highCrime']==1)/cleanCrime.highCrime.count())*100)
print("Negative Instances Percentage",(sum(cleanCrime['highCrime']==0)/cleanCrime.highCrime.count())*100)


#  Percentage of positive and negative instances Output:
# ********************************************************
# 
# Positive Intances count 1250
# Negative Instances count 743
# 
# Total Count 1993
# 
# Positive Instances Percentage 62.719518314099346
# Negative Instances Percentage 37.280481685900654

# ## Decision Tree

# In[125]:



from sklearn import tree
from sklearn.metrics import precision_score,recall_score,accuracy_score
crimeX = cleanCrime.drop(['highCrime', 'state','fold','ViolentCrimesPerPop','communityname'], axis=1)
crimeY= cleanCrime['highCrime']
clfDT =tree.DecisionTreeClassifier()
clfDT = clfDT.fit(crimeX, crimeY)

#crimeX.head()
#crimeY.head()

predictDT=clfDT.predict(crimeX)

y_actual = pd.Series(crimeY, name='Actual')
y_DTpredicted = pd.Series(predictDT, name='Predicted')
DT_confusion = pd.crosstab(y_actual, y_DTpredicted)
print("Confusion Matrix from Decision Tree with no Cross Validation")
print(DT_confusion)
print("Metrics")
precision = precision_score( y_actual, y_DTpredicted)
recall = recall_score( y_actual, y_DTpredicted)
accuracy= accuracy_score( y_actual, y_DTpredicted)
print("Precision = ",precision, ", Accuracy = ",accuracy,", Recall = ",recall)


# Decision Tree Output on entire dataset
# ***************************************
# Confusion Matrix:
# *****************
# Predicted    0     1
# 
# Actual             
# 
# 0          743     0
# 
# 1            0  1250
# 
# 
# Metrics:
# *******
# Precision =  1.0 
# Accuracy =  1.0 
# Recall =  1.0

# In[112]:


#Decision Tree Top Features

import matplotlib.pyplot as plt
from collections import defaultdict


importances = clfDT.feature_importances_
importances = importances[importances!=0]
indices = np.argsort(importances)[::-1]

topDTfeatures=defaultdict(lambda: len(indices))
for i in range(len(indices)):
 topDTfeatures[crimeX.columns[indices[i]]]=importances[indices[i]]
    
topDTfeatures=pd.DataFrame(list(topDTfeatures.items()),columns=['Feature Names','Coefficient'])
print("Top 10 Features used in Decision Tree")
print(topDTfeatures[:10])






# In[114]:


#cleanCrime['highCrime']
#cleanCrime.loc[cleanCrime['highCrime'] == 0, 'PctOccupManu']
print("\n Top Number 1 Feature")
print("\nPctOccupManu: percentage of people 16 and over and in manufacturing \n")
print("Average of PctOccupManu in Low Crime areas:",cleanCrime.loc[cleanCrime['highCrime'] == 0, 'PctOccupManu'].mean())
print("Average of PctOccupManu in High Crime areas:",cleanCrime.loc[cleanCrime['highCrime'] == 1, 'PctOccupManu'].mean())
print("The average of this feature is higher in high crime areas and hence it is an imporant feature ")

print("\n Top Number 2 Feature")
print ("racepctblack is the Percentage of population who are African American \n")
print ("Average of racepctblack for lowCrime = 0 is, ", cleanCrime.loc[cleanCrime['highCrime'] == 0, 'racepctblack'].mean())
print ("Average of racepctblack for highCrime = 1 is, ", cleanCrime.loc[cleanCrime['highCrime'] == 1, 'racepctblack'].mean())
print ("\nHigher the African american Perent in an Area's Population, higher the crime rate \n")

print("\n Top Number 3 Feature")
print ("racePctAsian is the Percentage of population who are Asiann \n")
print ("Average of racePctAsian for lowCrime = 0 is, ", cleanCrime.loc[cleanCrime['highCrime'] == 0, 'racePctAsian'].mean())
print ("Average of racePctAsian for highCrime = 1 is, ", cleanCrime.loc[cleanCrime['highCrime'] == 1, 'racePctAsian'].mean())
print ("\nHigher the Asian Perent in an Area's Population, higher the crime rate \n")



# The top 5 features of Decision Tree are 
# 
# 1.)PctOccupManu
# 2.)racepctblack  
# 3.)racePctAsian  
# 4.)NumUnderPov 
# 5.)blackPerCap     
# 
# ********************
# Top Number 1 Feature:
# 
# PctOccupManu: percentage of people 16 and over and in manufacturing 
# 
# Average of PctOccupManu in Low Crime areas: 0.3257604306864064
# Average of PctOccupManu in High Crime areas: 0.4301520000000005
# The average of this feature is higher in high crime areas and hence it is an imporant feature 
# 
# *********************
# 
# Top Number 2 Feature:
# 
# racepctblack is the Percentage of population who are African American 
# 
# Average of racepctblack for lowCrime = 0 is,  0.04769851951547786
# Average of racepctblack for highCrime = 1 is,  0.25740799999999986
# 
# Higher the African american Perent in an Area's Population, higher the crime rate 
# 
# *********************
# Top Number 3 Feature:
# 
# racePctAsian is the Percentage of population who are Asiann 
# 
# Average of racePctAsian for lowCrime = 0 is,  0.13242261103633923
# Average of racePctAsian for highCrime = 1 is,  0.16643200000000105
# 
# 
# Higher the Asian Perent in an Area's Population, higher the crime rate 
# 

# In[116]:


# Cross Validation in Decision Trees

from sklearn.model_selection import cross_val_score
cv_accuracy = cross_val_score(clfDT, crimeX, crimeY,cv=10,scoring="accuracy")
cv_recall = cross_val_score(clfDT, crimeX, crimeY,cv=10,scoring="recall")
cv_precision = cross_val_score(clfDT, crimeX, crimeY,cv=10,scoring="precision")
print("Cross Validation metric for Decision Trees")
print("CV Accuracy Mean:",cv_accuracy.mean())
print("CV Recall Mean",cv_recall.mean())
print("CV Prevision Mean",cv_precision.mean())
      
# When we fit the model on the same training and testing set, it overfits and hence we get perfect results with 100 percent accuracy
# But when we do cross validation we are predicting on out of sample data and hence its more reliable and has lower accuracy      
      


# Cross Validation in Decision Trees
# ********************************
# CV Accuracy Mean: 0.7284874371859298
# 
# 
# CV Recall Mean 0.764
# 
# 
# CV Prevision Mean 0.7892845796058703
# 

# ## Naive Bayes

# In[142]:


# Linear Classification - Gaussian NB Naive Bayes

from sklearn.naive_bayes import GaussianNB
clfNB = GaussianNB()
clfNB.fit(crimeX, crimeY)
predictNB=clfNB.predict(crimeX)

y_actual = pd.Series(crimeY, name='Actual')
y_NBpredicted = pd.Series(predictNB, name='Predicted')
NB_confusion = pd.crosstab(y_actual, y_NBpredicted)
print("Confusion Matrix from Naive Bayes with no Cross Validation")
print(NB_confusion)
print("Metrics")
precision = precision_score( y_actual, y_NBpredicted)
recall = recall_score( y_actual, y_NBpredicted)
accuracy= accuracy_score( y_actual, y_NBpredicted)
print("Precision = ",precision, ", Accuracy = ",accuracy,", Recall = ",recall)


# Confusion Matrix from Naive Bayes with no Cross Validation
# *********************************************************
# Predicted    0    1
# Actual             
# 
# 0          679   64
# 
# 1          378  872
# 
# 
# Metrics
# *******
# Precision =  0.9316239316239316 
# 
# Accuracy =  0.7782237832413447 
# 
# Recall =  0.6976

# In[143]:


# Cross Validation in Naive Bayes

from sklearn.model_selection import cross_val_score
cv_accuracy = cross_val_score(clfNB, crimeX, crimeY,cv=10,scoring="accuracy")
cv_recall = cross_val_score(clfNB, crimeX, crimeY,cv=10,scoring="recall")
cv_precision = cross_val_score(clfNB, crimeX, crimeY,cv=10,scoring="precision")
print("Cross Validation metrics for Naive Bayes")
print("Accuracy Mean",cv_accuracy.mean())
print("Recall Mean",cv_recall.mean())
print("Precision Mean",cv_precision.mean())


# Cross Validation metrics for Naive Bayes
# 
# Mean Accuracy : 0.761608
# Mean Recall:0.692
# Mean Precision:0.9117998148278733

# In[173]:


# Calculating top 10 most predictive features


from collections import OrderedDict
from operator import itemgetter


importances = clfDT.feature_importances_
importances = importances[importances!=0]
indices = np.argsort(importances)[::-1]



cleanCrime1= cleanCrime.drop(['state','fold','ViolentCrimesPerPop','communityname'], axis=1)
topNBfeatures=defaultdict(lambda: len(cleanCrime1.columns))
for col in cleanCrime1.columns:
    if(col!='highCrime'):
        colMeans=cleanCrime1.groupby('highCrime')[col].mean()    
        colVariance=cleanCrime1.groupby('highCrime')[col].var()
        score=(abs(colMeans[1]-colMeans[0]))/(np.sqrt(colVariance[1])+(np.sqrt(colVariance[0])))
        #print(score)
        topNBfeatures[col]=score
           


topNBfeatures = OrderedDict(sorted(topNBfeatures.items(), key=itemgetter(1),reverse=True)[:10])

topNBfeatures=pd.DataFrame(list(topNBfeatures.items()),columns=['Feature Names','Score'])
print("Top 10 Features used in Naive Bayes")
print(topNBfeatures[:10])
    


# 3.)The top 10 features used in NB are 
# 
# Top 10 Features used in Naive Bayes
# 
#       Feature Names     Score
#       
# 0       PctKids2Par  0.809336
# 
# 1        PctFam2Par  0.745162
# 
# 2      racePctWhite  0.734884
# 
# 3          PctIlleg  0.708929
# 
# 4      FemalePctDiv  0.693604
# 
# 5       TotalPctDiv  0.674282
# 
# 6  PctYoungKids2Par  0.664671
# 
# 7        pctWInvInc  0.660720
# 
# 8       PctTeen2Par  0.642621
# 
# 9    MalePctDivorce  0.616534
# 
# 
# 
# Using the formula makes sense to calculate predicted features because more the normalized difference between the 2 classes between they have a good margin between them and hence are good predictors.
# 
# Since Naive Bayes works on probabaility , having a good margin means the 2 classes will have significantly different probabailites and so classifying them will be easier
# 
# 4.) In comparison to Decision Trees
# 
# Decision Trees
# 
# Cross Validation metric for Decision Trees
# ************************************************
# CV Accuracy Mean: 0.7284874371859298
# 
# CV Recall Mean 0.764
# 
# CV Prevision Mean 0.7892845796058703
# 
# 
# Naive Bayes:
# ************
# Mean Accuracy : 0.761608
# 
# Mean Recall:0.692
# 
# Mean Precision:0.9117998148278733
# 
# 
# Naive Bayes has higher Accuracy
# Decision Tree has higher Recall
# Naive Bayes has higher Precision
# 
# 
# 

# ## Linear SVC

# In[176]:


### Linear SVC


from sklearn.svm import LinearSVC
clfSVC = LinearSVC(random_state=0)
clfSVC.fit(crimeX, crimeY)
predictSVC=clfSVC.predict(crimeX)

y_actual = pd.Series(crimeY, name='Actual')
y_SVCpredicted = pd.Series(predictSVC, name='Predicted')
SVC_confusion = pd.crosstab(y_actual, y_SVCpredicted)
print(SVC_confusion)
print("SVC Metrics")
precision = precision_score( y_actual, y_SVCpredicted)
recall = recall_score( y_actual, y_SVCpredicted)
accuracy= accuracy_score( y_actual, y_SVCpredicted)
print("Precision = ",precision, ", Accuracy = ",accuracy,", Recall = ",recall)


# SVM Training Set validation metrics
# ************************************
# Confusion Matrix:
# ******************
# Predicted    0     1
# Actual              
# 
# 0          599   144
# 
# 1          154  1096
# 
# SVM Metrics
# ***************
# Precision =  0.8838709677419355
# 
# Accuracy =  0.8504766683391871 
# 
# Recall =  0.8768
# 
# 
# 

# In[177]:


## Cross Validation using SVC

from sklearn.model_selection import cross_val_score
cv_accuracy = cross_val_score(clfSVC, crimeX, crimeY,cv=10,scoring="accuracy")
cv_recall = cross_val_score(clfSVC, crimeX, crimeY,cv=10,scoring="recall")
cv_precision = cross_val_score(clfSVC, crimeX, crimeY,cv=10,scoring="precision")
print("Cross Validation metrics for SVC")
print(cv_accuracy.mean())
print(cv_recall.mean())
print(cv_precision.mean())


# SVM Cross Validation validation metrics
# **************************************
# 
# Precision =  0.7962336683417085
# 
# Accuracy =  0.8343999999999999
# 
# Recall =  0.8454048565309501

# In[202]:


## Top most predictive features for SVM
from matplotlib import pyplot as plt

colnames=list(crimeX.columns.values)
abFeatureWeights=abs(clfSVC.coef_)[0]
weights=clfSVC.coef_[0]

topSVMfeatures=defaultdict(lambda: len(colnames))
for i in range(len(colnames)):
 topSVMfeatures[colnames[i]]=weights[i]

#print(topSVMfeatures)

topSVMfeatures=defaultdict(lambda: len(colnames))
for i in range(len(colnames)):
 topSVMfeatures[colnames[i]]=abFeatureWeights[i]

#print(topSVMfeatures)

topSVMfeatures = OrderedDict(sorted(topSVMfeatures.items(), key=itemgetter(1),reverse=True)[:10])



topSVMfeatures=pd.DataFrame(list(topSVMfeatures.items()),columns=['Feature Names','Coefficients'])
print("Top 10 Features used in SVM")
print(topSVMfeatures)


      


# Topmost predictive features of SVM
# 
# 
#       Feature Names  Coefficients
# 0        pctWInvInc      1.888488
# 1  PersPerOccupHous      1.755123
# 2      racePctWhite      1.500218
# 3       PctKids2Par      1.190327
# 4         RentHighQ      1.066884
# 5    MalePctDivorce      1.065697
# 6       NumUnderPov      1.051548
# 7         NumStreet      1.019154
# 8  PctOccupMgmtProf      1.014674
# 9        population      1.002300
# 
# Negatively Correlated Features with negative coeffiencets
# *********************************************************
# 1.) pctWInvInc: -1.88848775588396
#  
# This makes sense as more percentage of households with investment / rent income , it will tend to reduce Crime rate
# 
# 2.)racePctWhite: -1.500217861183296
#     
# This makes sense and means that percentage of white people leads to decreased crime
# 
# 3.) PctKids2Par    -1.190327177911603
# 
# This makes sense becuase percentage of kids in family housing with two parents means good family with family values and hence less prone 
# to crimes
#     
# Postively Correlated Features
# *****************************
# 1.)PersPerOccupHous      1.755123
# 
# Makes sense- more people in household means more crime
# 
# 2.) RentHighQ    1.0668844718981199
# 
# Does not make sense as areas with high rent means they are posher areas and should have low crime rate. Maybe people are stealing to pay
# their high rents
# 
# 3.)MalePctDivorce': 1.0656972776813411
# 
# Divorced men are more prone to commit crimes
# 
# 4.)NumUnderPov: 1.0515479694274288
#     
# Makes sense, more people under poverty , more crime is expected
#     
# 5.)NumStreet: 1.051548
#    
# Makes sense , more people on street , more crime
# 
# 6.)PctOccupMgmtProf': 1.0146737031345376
# 
# Does not make sense, people in professional occupations should commit less crie
# 
# 7.)population': 1.0023002868731028
# 
# More poulation  , more crime, makes sense
# 
# 
# Does calcualting Features  using coeffiecients make sense?
# 
#  Using coeffcients for feature selection makes sense as higher the co-efficient , it means they are more predictive  of the output
# 

# Comparison on SVM and Decision Trees
# **************************************
# 
# SVM has better precision, recall and acuracy than decision trees and hence is considered a better model fit for this dataset
# 

# #Linear Regression

# In[138]:


#Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

clfLR=LinearRegression()
Y_CrimePop = cleanCrime['ViolentCrimesPerPop']
clfLR.fit(crimeX,Y_CrimePop)
predictedValuesLR=clfLR.predict(crimeX)


# Mean Square Error on the Training Set
print("Mean Squared Error on the Training Set")
print(mean_squared_error(Y_CrimePop, predictedValuesLR))

# Cross Validation mean square error
lr_cv=cross_val_score(clfLR, crimeX, Y_CrimePop,cv=10,scoring="neg_mean_squared_error")
print ("Mean of MSE using Cross Validation: ", abs(lr_cv.mean()))

## Top Features predictive of high crime rate


LRcoef=list(zip(crimeX.columns,clfLR.coef_))
#sort(key=lambda x: x[1])
topPredLowCrime=sorted(LRcoef,key=lambda x: x[1])[:5]
print("\n\nTop Predictive Features of Low Crime")
print(topPredLowCrime)
topPredHighCrime=sorted(LRcoef,key=lambda x: -x[1])[:5]
print("\n\nTop Predictive Features of High Crime")
print(topPredHighCrime)

#


# Linear Regression Code Output:
# *****************************
# Mean Squared Error on the Training Set
# 0.016516774880307176
# 
# Mean of MSE using Cross Validation:  0.02009396930444532
# 
# 
# Top Predictive Features of Low Crime
# [('PctPersOwnOccup', -0.6756944788028595), ('TotalPctDiv', -0.5619243144146646), ('whitePerCap', -0.35101577444077936), ('PctKids2Par', -0.32265127649627473), ('OwnOccLowQuart', -0.30817021919318394)]
# 
# 
# Top Predictive Features of High Crime
# [('PersPerOccupHous', 0.6350881164986146), ('PctHousOwnOcc', 0.5681332098865255), ('MalePctDivorce', 0.45851704864166704), ('PctRecImmig8', 0.43251055765291024), ('MedRent', 0.3727277977113884)]
# 
# Linear Regression Code Output Explaination:
# ********************************************
# 
# 
# 1.)Estimate MSE of the Model using 10 fold Cross Validation : 0.02009396930444532
# 
# 
# 2.)Mean Square Error on Training Set : 0.016516774880307176
# 
# 
# 3.) Top 5 predictive feature of High Crime
# 
# a. )'PctHousOwnOcc'
# b.)'PersPerOccupHous'
# c.)'PersPerOccupHous'
# d.)'PctRecImmig8'
# e.)'MedRent'
# 
# 
# Top 5 predictive features of Low Crime
# 
# a.)'PctPersOwnOccup'
# b.)'TotalPctDiv'
# c.)'MalePctDivorce'
# d.)'whitePerCap'
# e.)'PctKids2Par'

# In[140]:


### Ridge Regression and pick best alpha out of (10, 1, 0.1, 0.01, and 0.001

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error

Y_CrimePop = cleanCrime['ViolentCrimesPerPop']

clfRidgeCV = RidgeCV(alphas=(10,1,0.1,0.01,0.001),cv=10)
clfRidgeCV.fit(crimeX,Y_CrimePop)
predictedValuesRidgeCV=clfRidgeCV.predict(crimeX)

print("Mean Squared Errorusing Ridge and 10 fold Cross Validation")
print(mean_squared_error(Y_CrimePop, predictedValuesRidgeCV))


clfRidge = RidgeCV(alphas=(10,1,0.1,0.01,0.001))
clfRidge.fit(crimeX,Y_CrimePop)
predictedValuesRidge=clfRidge.predict(crimeX)

print("Mean Squared Error on the Training Set using Ridge")
print(mean_squared_error(Y_CrimePop, predictedValuesRidge))


bestAlpha=clfRidge.alpha_
print("Best Alpha = ",bestAlpha)



# Ridge Regression Output
# ***********************
# 
# Mean Squared Error using Ridge and 10 fold Cross Validation
# 0.016763529155169477
# Mean Squared Error on the Training Set using Ridge
# 0.016763529155169432
# 
# Best Alpha =  1.0
# 
# Ridge Regression Output Explanation
# ***********************************
# 1.) Mean Squared Error using Ridge and 10 fold Cross Validation : 0.016763529155169477
# 2.) Mean Squared Error on the Training Set using Ridge :0.016763529155169432
# 3.) Best Alpha =  1.0
# 4.) Previously the MSE using Linear Regression  using 10 fold Cross Validation was  0.02, but now regularising using Ridge regression reduced the error down to 0.016763529155169477 and so we can conclude that by using ridge we reduced the weight and priority given to some parameters which would lead to Overfitting 
# 

# In[123]:


## Polynomial features for Regression

from sklearn.preprocessing import PolynomialFeatures

polyFeat = PolynomialFeatures(degree=2)
XPoly=polyFeat.fit_transform(crimeX)
clfPolyLR=LinearRegression()

## MSE from 10 fold cross validation

MSE_P=cross_val_score(clfPolyLR, XPoly, Y_CrimePop,cv=10,scoring="neg_mean_squared_error")
print("Mean MSE from 10 fold cross validation ", abs(MSE_P.mean()))

## MSE on training set

clfPolyLR.fit(XPoly,Y_CrimePop)
predictedValuesPoly=clfPolyLR.predict(XPoly)

print("Mean Squared Error on the Training Set using Ploynomial Features")
print(mean_squared_error(Y_CrimePop, predictedValuesPoly))



# ## Polynomial Features Output:
# ****************************
# 
# Mean MSE from 10 fold cross validation  0.1298981429166846
# Mean Squared Error on the Training Set using Polynomial Features
# 9.373499768284143e-29
# 
# Polynomial Features Output Explaination:
# ******************************************
# 1.)Mean MSE from 10 fold cross validation  0.1298981429166846
# 2.)Mean MSE training set : 2.645676919114767e-25
# 3.)The MSE from 10 fold cross validation using Polynomial features is 0.1298 which is higher than simple Linear regresssion(0.02) and ridge regression(0.0167) so we conclude using Polynomical features is not good for the dataset
# 
# 

# ## Dirty Data

# In[66]:


## Dirty Data:

dirtyCrime = pd.read_csv('./Crime Prediction Data/Crime Prediction Data/communities-crime-full.csv')
#print(dirtyCrime.head())
#print (dirtyCrime.dtypes)


# Decision Trees Unclean data
#a.	Create a new field “highCrime” which is true if the crime rate per capita (ViolentCrimesPerPop) is greater than 0.1
dirtyCrime['highCrime']=(dirtyCrime['ViolentCrimesPerPop'] > 0.1).astype(int)
dirtyCrime.head()


# Percentage of positive and negative instances
#Count
print("Positive Intances count",sum(dirtyCrime['highCrime']==1))
print("Negative Instances count",sum(dirtyCrime['highCrime']==0))
print("Total Count",dirtyCrime.highCrime.count())
#Percentage
print("Positive Instances Percentage",(sum(dirtyCrime['highCrime']==1)/dirtyCrime.highCrime.count())*100)
print("Negative Instances Percentage",(sum(dirtyCrime['highCrime']==0)/dirtyCrime.highCrime.count())*100)


# Decision Tree output Part 1
# ******************************
# Positive Intances count 1251
# Negative Instances count 743
# 
# Total Count 1994
# 
# 
# Positive Instances Percentage 62.7382146439318
# Negative Instances Percentage 37.2617853560682

# In[135]:




# Removing unwanted columns such as State, Fold, Community name and number
# Imputing missing value with mean
 
from sklearn import tree
from sklearn.metrics import precision_score,recall_score,accuracy_score
dirtycrimeX = dirtyCrime.drop(['highCrime','fold','county','community','state', 'ViolentCrimesPerPop','communityname'], axis=1)
dirtycrimeY= dirtyCrime['highCrime']

dirtycrimeX=dirtycrimeX.apply(pd.to_numeric,errors='coerce')
dirtycrimeX = dirtycrimeX.fillna(dirtycrimeX.mean()).astype(int).astype(int)


#print(dirtycrimeX.head())
#print(dirtycrimeY.head())


clfDT_dirty =tree.DecisionTreeClassifier()
clfDT_dirty = clfDT_dirty.fit(dirtycrimeX, dirtycrimeY)



predictDT_dirty=clfDT_dirty.predict(dirtycrimeX)

y_actualD = pd.Series(dirtycrimeY, name='Actual')
y_DTpredictedD = pd.Series(predictDT_dirty, name='Predicted')
DT_confusionD = pd.crosstab(y_actualD, y_DTpredictedD)
print("Confusion Matrix from Decision Tree with Unclean data with no Cross Validation")
print(DT_confusionD)
print("Metrics")
precision = precision_score( y_actualD, y_DTpredictedD)
recall = recall_score( y_actualD, y_DTpredictedD)
accuracy= accuracy_score( y_actualD, y_DTpredictedD)
print("Precision = ",precision, ", Accuracy = ",accuracy,", Recall = ",recall)


# 
# Confusion Matrix from Decision Tree with Unclean data with no Cross Validation
# **********************************************************************************
# 
# Predicted    0     1
# Actual              
# 
# 0          239   504
# 
# 1           51  1200
# 
# 
# 
# Metrics
# *******
# Precision =  0.704225352112676 
# 
# Accuracy =  0.7216649949849548
# 
# Recall =  0.9592326139088729

# In[136]:


## Decision tree on Dirty Dat - Top Features

#Decision Tree Top Features
import matplotlib.pyplot as plt
from collections import defaultdict


importances = clfDT_dirty.feature_importances_
importances = importances[importances!=0]
indices = np.argsort(importances)[::-1]

topDTfeatures=defaultdict(lambda: len(indices))
for i in range(len(indices)):
 topDTfeatures[dirtycrimeX.columns[indices[i]]]=importances[indices[i]]
    
topDTfeatures=pd.DataFrame(list(topDTfeatures.items()),columns=['Feature Names','Coefficient'])
print("Top 10 Features used in Decision Tree for Dirty Data")
print(topDTfeatures[:10])


# The top most predictive features are as below in the Dirty Data
# 0           agePct65up     0.091439
# 1         PctImmigRec8     0.063288
# 2        householdsize     0.054769
# 3      PctPersOwnOccup     0.045037
# 4        NumInShelters     0.038777
# 
# 
# They make sense because
# agePct65up - could be that older people are easy targets for crime
# 
# PctImmigRec8 - Immigrants are more prone to perform criminal activties
# 
# householdsize - more people in the house, means they are more prone to crime
# 
# PctPersOwnOccup -percent of people in owner occupied households 
# 
# NumInShelters - homeless people more prone to criminal activities

# In[137]:


# Cross Validation in Decision Trees on Dirty Data

from sklearn.model_selection import cross_val_score
cv_accuracy = cross_val_score(clfDT_dirty, dirtycrimeX, dirtycrimeY,cv=10,scoring="accuracy")
cv_recall = cross_val_score(clfDT_dirty, dirtycrimeX, dirtycrimeY,cv=10,scoring="recall")
cv_precision = cross_val_score(clfDT_dirty, dirtycrimeX, dirtycrimeY,cv=10,scoring="precision")
print("Cross Validation metric for Decision Trees for Drity Data")
print("CV Accuracy Mean:",cv_accuracy.mean())
print("CV Recall Mean",cv_recall.mean())
print("CV Prevision Mean",cv_precision.mean())
      
# When we fit the model on the same training and testing set, it overfits and hence we get perfect results with 100 percent accuracy
# But when we do cross validation we are predicting on out of sample data and hence its more reliable and has lower accuracy      
      


# 
# Dirty CV Values:
# ********************
# Cross Validation metric for Decision Trees for Dirty Data
# 
# CV Accuracy Mean: 0.6700034500862522
# CV Recall Mean 0.9176634920634921
# 
# CV Prevision Mean 0.6733723383174538
# 
# Previous CV values
# *********************
# 
# Cross Validation metric for Decision Trees
# 
# CV Accuracy Mean: 0.7284874371859298
#     
# CV Recall Mean 0.764
# 
# CV Prevision Mean 0.7892845796058703
# 
# 
# The Cross Validation metrics are atleast 10 percent lower for Dirty data , so cleaning Data is important. however recall is higher on dirty data
