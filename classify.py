import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np

# functions to change data to numerical (changing attributes such as color to numbers)
df = pd.read_csv("mushrooms.csv")
def convert_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))
    return df

df = convert_data(df)
x = df.drop(columns=["class"])
y = df["class"]

# splitting between training set and testing set 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = tf.keras.models.Sequential() 
# sigmoid function is function from 0-1, helps because diagnosis is from 0-1
model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation="sigmoid")) 
model.add(tf.keras.layers.Dense(256, activation="sigmoid"))
model.add(tf.keras.layers.Dense(1, activation="sigmoid")) # outer layer

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)