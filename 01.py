import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dfS = pd.read_csv("train.csv")
dfT = pd.read_csv("test.csv")

n1 = dfS['Age'].isnull().values.sum() 
n2 = dfS['Age'].isnull().values.sum() / len(dfS['Age']) * 100

#print(n1)
#print(n2)

#plt.hist(dfS['Age'].dropna(),bins=20)
#plt.show()

mean = np.mean(dfS['Age'])
dfS['Age'] = dfS['Age'].fillna(mean)

dfS['Sex'] = dfS['Sex'].str.replace('female','2')
dfS['Sex'] = dfS['Sex'].str.replace('male','1')

# トレーニングデータを説明変数(X)と目的変数(y)に分割
X = pd.DataFrame({     'Pclass':dfS['Pclass'],
                       'Sex':dfS['Sex'],
                       'Age':dfS['Age']})
y = pd.DataFrame({'Survived':dfS['Survived']})

# 学習用データと検証用データに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None )

# 学習
model = SVC(kernel='linear', random_state=None,C=0.1)
model.fit(X_train, y_train)

# 精度検証
score = model.score(X_test,y_test)
print(score)

### テストデータの前処理
mean = np.mean(dfT['Age'])
dfT['Age'] = dfT['Age'].fillna(mean)

dfT['Sex'] = dfT['Sex'].str.replace('female','2')
dfT['Sex'] = dfT['Sex'].str.replace('male','1')

Xtest = pd.DataFrame({     'Pclass':dfT['Pclass'],
                       'Sex':dfT['Sex'],
                       'Age':dfT['Age']})

# 予測
result = model.predict(Xtest)
print(result)

# データの整形
submitPre = pd.DataFrame({
                        'PassengerId':dfT['PassengerId'],
                        'Survived':result
                        })
# CSV出力
submitPre.to_csv("gender_submission.csv",index=False)