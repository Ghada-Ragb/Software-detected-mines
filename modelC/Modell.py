import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import pickle
import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv("data1.csv")
encoder=LabelEncoder()
data['diagnosis']=encoder.fit_transform(data['diagnosis'])


X = data.iloc[:, 2:9].values
y = data.iloc[:, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model=SVC(kernel = 'linear', random_state = 0)
model.fit(X_train, y_train)

scores=-1*cross_val_score(model,X,y,cv=7,scoring="neg_mean_absolute_error")
print(scores)
Y_Pred=model.predict(X_test)
score = accuracy_score(y_test, Y_Pred)
print(score)
pickle.dump(model,open('model.pkl','wb'))