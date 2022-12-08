from fastapi import FastAPI, UploadFile
from typing import List
import numpy as np
from tensorflow.keras.preprocessing import timeseries_dataset_from_array
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'hi': True}

@app.post("/predict/")
async def create_upload_file(files: List[UploadFile]):

    content = await files[0].read()
    X_train = np.frombuffer(content, dtype=np.float32)
    X_train = X_train.reshape(int(len(X_train)/3), 3)


    content = await files[1].read()
    y_train = np.frombuffer(content, dtype=np.float32)
    #model + fit#
    dataset_test = timeseries_dataset_from_array(
        X_train,
        y_train,
        sequence_length=50,
        batch_size=32,
    )
    model = Sequential()
    model.add(layers.LSTM(units=32, input_shape=(50,3)))
    model.add(layers.Dense(1, activation="linear"))

    model.compile(loss='mse',
                optimizer='adam')

    model.fit(dataset_test,
         epochs=150, verbose=0)
    pred = model.predict(dataset_test)
    return {'pred': np.array(pred).tolist()}
