import pandas as pd
import numpy as np
from sklearn import linear_model
df1 = pd.read_csv(r"C:\Users\user\Desktop\PES\Arithmenia-linear_regression\soildata.csv")
reg = linear_model.LinearRegression()
reg.fit(df1[['Temperature','Humidity','Moisture']],df1.N)
a=reg.predict([[28,52,40]])
print("The standard content of nitrogen(kg/ha) needed in a banana crop is 500")
print("Content of nitrogen(kg/ha) needed in your fertilizer is: ",500-a)
reg.fit(df1[['Temperature','Humidity','Moisture']],df1.P)
b=reg.predict([[28,52,40]])
print("The standard content of phosphorous(kg/ha) needed in a banana crop is 150")
print("Content phosphorous(kg/ha) needed in your fertilizer is: ",150-b)
reg.fit(df1[['Temperature','Humidity','Moisture']],df1.K)
c=reg.predict([[28,52,40]])
print("The standard content of potassium(kg/ha) needed in a banana crop is 350")
print("Content of potassium(kg/ha) needed in your fertilizer is: ",350-c)
print("\n")
print(a,b,c)





