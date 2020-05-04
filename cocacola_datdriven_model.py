# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:47:35 2020

@author: tejas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.graphics.tsaplots as tsaplots
from datetime import datetime,time
coca=pd.read_excel("D:\TEJAS FORMAT\EXCELR ASSIGMENTS\COMPLETED\FORECASTING\COCACOLA SALES\CocaCola_Sales_Rawdata.xlsx")
coca.isnull().sum()
import seaborn as sns
sns.boxplot("Sales",data=coca)

for i in range(2,24,6):
    coca["Sales"].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=4)
import statsmodels.api as sm
seasonal_dec=sm.tsa.seasonal_decompose(coca["Sales"],freq=3)
seasonal_dec.plot()

train=coca.head(38)
test=coca.tail(4)
test=test.set_index(np.arange(1,5))

def MAPE(pred,org):
    temp=np.abs((pred-org))*100/org
    return np.mean(temp)

######Simple Exp#######
Exp=SimpleExpSmoothing(train["Sales"]).fit()
Exp_pred=Exp.predict(start=test.index[0],end=test.index[-1])
Exp_mape=MAPE(Exp_pred,test.Sales)########54.41

##Holts#########
hw=Holt(train["Sales"]).fit()
hw_pred=hw.predict(start=test.index[0],end=test.index[-1])
hw_mape=MAPE(hw_pred,test.Sales)##55.09

####Holts winter Exp smoothing with additive trend and additive seasonality#############
Exp_add_add=ExponentialSmoothing(train["Sales"],damped=True,seasonal="add",seasonal_periods=4,trend="add").fit()
Exp_add_add_pred=Exp_add_add.predict(start=test.index[0],end=test.index[-1])
Exp_add_add_Mape=MAPE(Exp_add_add_pred,test.Sales)#####52.01

#####Holts winter Exp smoothing Multiplicative trend with  add seasonality#########
Exp_mul_add=ExponentialSmoothing(train["Sales"],damped=True,seasonal="mul",seasonal_periods=4,trend="add").fit()
Exp_mul_add_pred=Exp_mul_add.predict(start=test.index[0],end=test.index[-1])
Exp_mul_add_mape=MAPE(Exp_mul_add_pred,test.Sales)######53.29


# Visualization of Forecasted values for Test data set using different methods
plt.plot(train.index,train["Sales"],label="Train",color="black") 
plt.plot(test.index,test["Sales"],label="Test",color="blue")
plt.plot(Exp_pred.index,Exp_pred,label="Simple Exp Smoothing",color="yellow")
plt.plot(hw_pred.index,hw_pred,label="Holts method",color="orange")
plt.plot(Exp_add_add_pred.index,Exp_add_add_pred,label="Exp Smoothing with add trend & add seasonality",color="blue")
plt.plot(Exp_mul_add_pred.index,Exp_mul_add_pred,label="Exp Smoothing with add trend & mul seasonality",color="violet")

####Storing all mape values##########
Table={"Model":pd.Series(["Exp_mape","hw_mape","Exp_add_add_Mape","Exp_mul_add_mape"]),"MAPE VAlUES":pd.Series([Exp_mape,hw_mape,Exp_add_add_Mape,Exp_mul_add_mape])}
Table=pd.DataFrame(Table)     
