# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:36:28 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as snf
Cocacola=pd.read_excel("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\COCACOLA SALES\CocaCola_Sales_Rawdata.xlsx")
Cocacola.head()
Cocacola.isnull().sum()
quarter=["Q1,Q2,Q3,Q4"]
p=Cocacola["Quarter"][0]
p[0:2]
Cocacola["sales"]=0

for i in range(42):
    p=Cocacola["Quarter"][i]
    Cocacola["sales"][i]=p[0:2]
    
quarter_dummies=pd.DataFrame(pd.get_dummies(Cocacola["sales"]))
Coca=pd.concat([Cocacola,quarter_dummies],axis=1)

Coca["t"]=np.arange(1,43)
Coca["t_square"]=Coca["t"]*Coca["t"]
Coca["log_sales"]=np.log(Coca["Sales"])



Train=Coca.head(38)
Test=Coca.tail(4)
Test=Test.set_index(np.arange(1,5))

######Linear model##########
Lin_model=snf.ols("Sales~t",data=Train).fit()
Lin_pred=pd.Series(Lin_model.predict(pd.DataFrame(Test["t"])))
Lin_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Lin_pred))**2))###########591.55

#####Quadratic Model##########
Quad_model=snf.ols("Sales~t+t_square",data=Train).fit()
Quad_pred=pd.Series(Quad_model.predict(pd.DataFrame(Test[["t","t_square"]])))
Quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Quad_pred))**2))###########475.56

####Exp Model#####
Exp_model=snf.ols("log_sales~t",data=Train).fit()
Exp_pred=pd.Series(Exp_model.predict(pd.DataFrame(Test["t"])))
Exp_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Exp_pred)))**2))###########466.24

###Building the Additive seasonality############
Add_sea=snf.ols("Sales~Q1+Q2+Q3+Q4",data=Train).fit()
Add_pred=pd.Series(Add_sea.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4"]])))
Add_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_pred))**2))###########1860.02

###Building the Additive seasonality with linear trend############
Add_lin_sea=snf.ols("Sales~Q1+Q2+Q3+Q4+t",data=Train).fit()
Add_lin_pred=pd.Series(Add_lin_sea.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4","t"]])))
Add_lin_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_lin_pred))**2))###########462.98

###Building the Additive with Quadratic trend######
Add_Quad_sea=snf.ols("Sales~Q1+Q2+Q3+Q4+t+t_square",data=Train).fit()
Add_Quad_pred=pd.Series(Add_Quad_sea.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4","t","t_square"]])))
Add_Quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(Add_Quad_pred))**2))###########31.73

###Building the Multiplicative model###########
Mul_model=snf.ols("log_sales~Q1+Q2+Q3+Q4",data=Train).fit()
Mul_pred=pd.Series(Mul_model.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4"]])))
Mul_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_pred)))**2))###########1963.38

###Building Multiplicative with linear trend############
Mul_lin_model=snf.ols("log_sales~Q1+Q2+Q3+Q4+t",data=Train).fit()
Mul_lin_pred=pd.Series(Mul_lin_model.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4","t"]])))
Mul_lin_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_lin_pred)))**2))###########225.52

###Building Multiplicative with linear Quadratic trend############
Mul_lin_quad_model=snf.ols("log_sales~Q1+Q2+Q3+Q4+t+t_square",data=Train).fit()
Mul_lin_quad_pred=pd.Series(Mul_lin_quad_model.predict(pd.DataFrame(Test[["Q1","Q2","Q3","Q4","t","t_square"]])))
Mul_lin_quad_rmse=np.sqrt(np.mean((np.array(Test["Sales"])-np.array(np.exp(Mul_lin_quad_pred)))**2))###########581.84

###Table for rmse values########
Rmse_Table={"Model":pd.Series(["Lin_rmse","Quad_rmse","Exp_rmse","Add_rmse","Add_lin_rmse","Add_Quad_rmse","Mul_rmse","Mul_lin_rmse","Mul_lin_quad_rmse"]),"RMSE_Values":pd.Series([Lin_rmse,Quad_rmse,Exp_rmse,Add_rmse,Add_lin_rmse,Add_Quad_rmse,Mul_rmse,Mul_lin_rmse,Mul_lin_quad_rmse])}
Rmse_Table=pd.DataFrame(Rmse_Table)

##From above RMSE values Multiplicative with linear trend is having less#######
Final=Mul_lin_model=snf.ols("log_sales~Q1+Q2+Q3+Q4+t",data=Coca).fit()
