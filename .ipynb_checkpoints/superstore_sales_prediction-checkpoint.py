import numpy as np
import pandas as pd

df = pd.read_csv('Superstore.csv', encoding = 'windows-1252')

df.info()

df = df.drop(['Row ID', 'Country', 'Product ID', 'Product Name', 'Order ID', 'Customer Name','Customer ID'], axis = 1)

def encode_dates(df, column):
    df = df.copy()
    df[column] = pd.to_datetime(df[column], format='mixed')
    df[column + '_year'] = df[column].apply(lambda x: x.year)
    df[column + '_month'] = df[column].apply(lambda x: x.month)
    df[column + '_day'] = df[column].apply(lambda x: x.day)
    df = df.drop(column, axis=1)
    return df

def onehot_encode(df, column):
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(column, axis=1)
    return df

df = encode_dates(df, column="Order Date")
df = encode_dates(df, column='Ship Date')


df['Ship Mode'].unique()


for column in ['Ship Mode','Segment','City','State','Postal Code','Region','Category','Sub-Category']:
  df = onehot_encode(df, column=column)


X = df.drop('Sales', axis=1)
Y = df['Sales']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
X_train.shape

X_train.describe()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns= X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X.columns)

import tensorflow as tf
from keras.layers import Dense
inputs = tf.keras.Input(shape =(X_train.shape[1],))
x = tf.keras.layers.Dense(128, activation='relu')(inputs)
x = tf.keras.layers.Dense(1251, activation='relu')(x)
outputs = tf.keras.layers.Dense(1, activation='linear')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile(optimizer='adam', loss='mse')
history = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test,Y_test),
    batch_size=32,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau()
    ]
)

test_loss = model.evaluate(X_test,Y_test, verbose=0)
print("Test Loss: {:.5f}".format(test_loss))

from sklearn.metrics import r2_score
y_pred = np.squeeze(model.predict(X_test))
test_r2 = r2_score(Y_test, y_pred)
print("Test R^2 Score: {:.5f}".format(test_r2))

import pickle
pickle.dump(model, open('model.pkl', 'wb'))

