import pydicom
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.optimizers import Adam
from keras_contrib.layers import InstanceNormalization
from typing import List

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageFilter
from datetime import datetime, date
import os, uuid, requests, psycopg2
from typing import Union

optimizer = Adam(0.0002, 0.5)
# load json and create model
json_file = open('little_generator.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
little_model = model_from_json(loaded_model_json, custom_objects={'InstanceNormalization':InstanceNormalization})
# load weights into new model
little_model.load_weights("little_generator_weights.hdf5")
 
# evaluate loaded model on test data
little_model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

json_file = open('big_generator.json', 'r')
big_model_json = json_file.read()
json_file.close()
big_model = model_from_json(big_model_json, custom_objects={'InstanceNormalization':InstanceNormalization})
# load weights into new model
big_model.load_weights("big_generator_weights.hdf5")
 
# evaluate loaded model on test data
big_model.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

app = FastAPI()

print(os.getcwd())

@app.put("/image/")
async def generate_pathology(
	image_url: str, pathology_size: int, x: int, y: int, new_image_url: str
):
	path = image_url
	data = pydicom.dcmread(path)
	image = data.pixel_array
	if pathology_size <= 10:
		radius = 16
	else:
		radius = 32
	square = image[max(0, y - radius):min(image.shape[0], y + radius), max(0, x - radius):min(image.shape[1], x + radius)]
	img = np.expand_dims(square, axis=0)
	img = np.expand_dims(img, axis=3)
	if pathology_size <= 10:
		gen_square = little_model.predict(img)
	else:
		gen_square = big_model.predict(img)
	gen_square = np.squeeze(gen_square, axis=3)
	gen_square = np.squeeze(gen_square, axis=0)
	image[max(0, y - radius):min(image.shape[0], y + radius), max(0, x - radius):min(image.shape[1], x + radius)] = gen_square
	data.PixelData = image.tobytes()
	data.save_as('image.dcm', write_like_original=False)
	return {}
