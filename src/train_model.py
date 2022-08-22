from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pickle import load, dump

X,y = load(open('../dataset/datasets_v001.pkl', 'rb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

pipeline = Pipeline(
    [('scaler', StandardScaler()),
     ('classifier', DecisionTreeClassifier())
     ]
)

pipeline.fit(X_train,y_train)

score = pipeline.score(X_test, y_test)

dump(pipeline, open('../model/model_v001.pkl', 'wb'))





