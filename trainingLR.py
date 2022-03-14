import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
#đọc dữ liệu
df = pd.read_csv("train1.csv")
dft = pd.read_csv("test1.csv")
#Kiểm tra dữ liệu cột chất lượng:
df['quality'].value_counts()

#Thông tin các dữ liệu train:
df.info()

#Bản đồ tương quan:
plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(),annot=True)

#Tạm bỏ cột ID:
df.drop(["Id"],axis = 1,inplace = True)
ids = dft['Id']
dft.drop(["Id"],axis = 1,inplace = True)

print(df)

print("Dữ liệu cần kiểm tra:\n")
print(dft)

#Build model
#PP: Hồi quy tuyến tính:
X = df.drop( "quality",axis=1)
y = df["quality"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LogisticRegression()
lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

#Kiểm tra trên tệp dữ liệu train:
dz = pd.DataFrame({'Gt chính xác': y_test, 'Dự đoán': y_pred})
print(dz)

#Kiểm tra tỉ lệ chính xác:
print('Độ chính xác: ')
accuracy_score(y_test,y_pred)
#Lưu model lại:
import pickle
filename = 'finalized_model.sav'
pickle.dump(lr, open(filename, 'wb'))

#Kiểm tra trên tệp dữ liệu test:
test = lr.predict(dft)
Predf = pd.DataFrame(columns=["Id",'quality'])
Predf['Id']= ids
Predf['quality']= test
print(Predf)

Predf.to_csv('Solution.csv',index=False)