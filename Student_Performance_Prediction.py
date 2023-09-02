# Importing Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing The Dataset
dataset = pd.read_csv('./Data/student-mat.csv', sep=';')
X = dataset.iloc[:, 0:32]
X2 = dataset.iloc[:, 0:32]
y = dataset.iloc[:, 32]




# Encoding Categorical Data
labelEncoder_X = LabelEncoder()

variables_to_labelEncode = [0, 1, 3, 4, 5, 8, 9 ,10 , 11, 15, 16, 17, 18, 19, 20, 21, 22]

for i in variables_to_labelEncode:
    X.iloc[:, i] = labelEncoder_X.fit_transform(X.iloc[:, i])

# One Hot Encoding
oneHotEncoder_X = OneHotEncoder(sparse = False)

X['Mjob_at_home'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 0]
X['Mjob_health'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 1]
X['Mjob_other'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 2]
X['Mjob_services'] = oneHotEncoder_X.fit_transform(X[['Mjob']])[:, 3]

X['Fjob_at_home'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 0]
X['Fjob_health'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 1]
X['Fjob_other'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 2]
X['Fjob_services'] = oneHotEncoder_X.fit_transform(X[['Fjob']])[:, 3]

X['reason_course'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 0]
X['reason_other'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 1]
X['reason_home'] = oneHotEncoder_X.fit_transform(X[['reason']])[:, 2]

X['guardian_father'] = oneHotEncoder_X.fit_transform(X[['guardian']])[:, 0]
X['guardian_mother'] = oneHotEncoder_X.fit_transform(X[['guardian']])[:, 1]

# Drop old columns
X.drop(['Mjob', 'Fjob', 'reason', 'guardian'],axis='columns', inplace=True)

# Reorder columns
X = X[['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 
      'Mjob_at_home', 'Mjob_health', 'Mjob_other', 'Mjob_services',
      'Fjob_at_home', 'Fjob_health', 'Fjob_other', 'Fjob_services',
      'reason_course', 'reason_other', 'reason_home', 'guardian_father',
       'guardian_mother','traveltime', 'studytime', 'failures', 'schoolsup', 
       'famsup', 'paid',
       'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel',
       'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]


# Dimensionality Reduction
variance_vector = X.var()

correlation_matrix = X.corr()

# Drop some variables due to high correlation between them and other variables

    # Fedu and Medu => according to this article (https://www.theguardian.com
    # /society/2014/sep/23/fathers-education-child-success-school) 
    # father's education is more important for children than mother's 
    # so i chose to delete Medu
    
    # Walc and Dalc => i chose to drop Walc because drinking in workday has more impact
    # than week end
    
    # G1 and G2 => and of course i dropped  first semester's grades because
    # the second semester is more close the the third semester, 
    # thus it could be more impactful

X.drop(['Medu', 'Walc', 'G1'],axis='columns', inplace=True)




# Using Random Forest Feature importance to select the most important features
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(random_state=1, max_depth=10)

model.fit(X,y)

features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-1:-12:-1]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

X = X.iloc[:, indices]

#from sklearn.feature_selection import SelectFromModel
#feature = SelectFromModel(model)
#Fit = feature.fit_transform(X, y)

# Splitting the data to train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler

X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

y_scaler = StandardScaler()
y_train = np.array(y_train).reshape(-1, 1)
y_train = y_scaler.fit_transform(y_train)


# Using Principal Component Analysis to extract the most important features

from sklearn.decomposition import PCA

pca = PCA(n_components=10)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.bar(range(len(pca.explained_variance_ratio_)), np.cumsum(pca.explained_variance_ratio_))

#from sklearn.decomposition import TruncatedSVD 
#svd = TruncatedSVD(n_components=4)
#svd_result = svd.fit_transform(X.values)
#plt.bar(range(4), svd.explained_variance_ratio_)


# Train a Support Vector Regression (WITH RBF Kernel) model on the preprocessed data

from sklearn import svm
svr = svm.SVR()
svr.fit(X_train, y_train.ravel())
y_pred = svr.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred, squared=False)



# Using a Simple linear regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_pred = y_scaler.inverse_transform(y_pred)

rmse = mean_squared_error(y_test, y_pred, squared=False)

#### (feature selection based on multicollinearity and random forest feature importance)




############################## CONCLUSIONS ###################################

y_pred_2=np.squeeze(y_pred)
df=pd.DataFrame({'Actual': y_test,'Prediction': y_pred_2})
df
plt.plot(range(len(y_pred)), y_pred, color = 'red')
plt.plot(range(len(y_test)), y_test, color = 'blue')
plt.title('Predicted G3 VS Real G3')
plt.xlabel('Students')
plt.ylabel('Grades')
plt.show()

















