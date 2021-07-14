import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



def print_hi(name):

    print(f'Hi, {name}')



if __name__=='__main__':
    print_hi('PyCharm')

    dataset = pd.read_csv('Churn_Modelling.csv')
    X = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1].values
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:,2])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    ann = tf.keras.Sequential()
    #adding the input layer or the first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    #adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    #adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

    #compiling the ann

    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ann.fit(X_train,y_train, batch_size=32, epochs=100)

    print(ann)



