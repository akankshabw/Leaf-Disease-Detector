from fastapi import FastAPI, File, UploadFile
from enum import Enum
import uvicorn 
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"]
)

MODEL = tf.keras.models.load_model("../models/1.keras")
class_names = ["Early Blight", "Late Blight", "Healthy"]


@app.get("/ping")
async def ping():
    return "Hello, How is Life !!"


def read_file_as_image(data):
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    prediction = MODEL.predict(img_batch)
    predicted_label = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    return {
        'class': predicted_label,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app,host='localhost', port=8000)




######## OVERVIEW ###############
# class AvailableCuisines(str, Enum):
#     indian = "indian"
#     american = "american"
#     italian = "italian"


# food_items ={
#     'indian' : ["samosa, chaat"],
#     'american' : ["Hot dog", "Apple pie"],
#     'italian' : ["Ravioli", "Pizza"] 
# }

# @app.get("/get_items/{cuisine}")
# async def get_items(cuisine: AvailableCuisines):
#     return food_items.get(cuisine)


# @app.get("/hello/{name}")
# async def hello(name):
#     return f"welcome to my python webpage, {name}!" 
#######################################
