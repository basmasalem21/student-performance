#%%--------------------------------------------------
#%%import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
#%%----------------------------------------------------
#%%----------------------------------------------------
# إعدادات الطباعة
pd.set_option('display.max_columns', None)  # إظهار جميع الأعمدة
pd.set_option('display.max_rows', None)     # إظهار جميع الصفوف (إن لزم الأمر)
pd.set_option('display.float_format', '{:.2f}'.format)  # تقليل الأرقام العشرية (اختياري)
#%%----------------------------------------------------
#import dataset:-
dataset_path="C:\\Users\\Elhussien\\Desktop\\machine learning\\Student_Performance.csv"
dataset=pd.read_csv(dataset_path)
print(dataset.info())
#print(dataset.isnull().sum())   #0
#%%------------------------------------------------------
#%%print corrolation and draw its graph
dataset2=dataset.copy()
dataset2=dataset2.drop('Extracurricular Activities',axis=1)
print(dataset2.info())
corr=dataset2.corr()
print(corr)
sns.heatmap(corr,annot=True)
plt.show()

#%%---------------------------------------------------------------
#%%drawing histogram:-
dataset2.hist(figsize=(20,30))
plt.show()
#%%--------------------------------------------------------------
#%%describing data:-
print(dataset2.describe().T)
#%%---------------------------------------------------------------
#%%drawing first 10 students data
df1 = dataset2.head(10)
df1.plot(kind="bar", figsize=(16,10))
plt.grid(which="major",linestyle="-", linewidth="0.5",color="green")

plt.show()
#%%-----------------------------------------------------------------
print(dataset.info())
#%%--------------------------------------------------------------
#onehot encoding:-
new_column=pd.get_dummies(dataset['Extracurricular Activities'],dtype=int,drop_first=True)
dataset=pd.concat([dataset,new_column],axis=1)
dataset.drop('Extracurricular Activities',axis=1,inplace=True)
print(dataset.head())
#%%-----------------------------------------------------------------
#%%determining x and y:-
x=dataset.drop('Performance Index',axis=1)
y=dataset['Performance Index']
#print(x.shape)
#print(y.shape)
y=y.values.reshape(-1,1)
#print(y.shape)
#%%------------------------------------------------------------------
import statsmodels.api as sm
stmodel1=sm.OLS(y,x).fit()
print(stmodel1.summary())

#%%------------------------------------------------------------------
#8888888888888888888888888888888888888888888888888888888888888888888888888
# #%%splitting data into train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=30)
#%%----------------------------------------------
#%%linear regression model
my_model=LinearRegression()
my_model.fit(x_train,y_train)
#%%----------------------------------------------
#%% prediction using model
y_pred=my_model.predict(x_test)
#%%---------------------------------------------
#%%converting array to series for putting them in the dataframe
y_test = y_test.squeeze()  # تحويل إلى Series إذا كانت DataFrame
y_pred = y_pred.squeeze()  # تحويل إلى Series إذا كانت DataFrame
#%%---------------------------------------------
#%%dataframe for y_test,y_pred,error btw them
dataframe=pd.DataFrame({"True":y_test,
    "predicted":y_pred,
    'error':abs(y_test-y_pred)
})
print(dataframe)
#%%---------------------------------------------
#%%printing mse  and r2 score
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"""
      Mean Square Error is {np.sqrt(mse)}
      R2 Score is {r2}
""")
#%%--------------------------------------------
#%%printing coeficient and intercept of the model
print(f"model intercept is : {my_model.intercept_}")
print(f"model coeficient is : {my_model.coef_}")
#print(f"equation is ")

















