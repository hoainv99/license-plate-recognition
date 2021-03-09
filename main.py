from pydantic import BaseModel

import secrets
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi import File, UploadFile

from plate_detection.detector import Predictor
from plate_detection.utils import draw_coordinate, image_array_to_base64,draw_box
from config import *
import cv2
import os
import shutil
import numpy as np

from plate_recognition.recognizer import Recognizer

# detecter = DetectLP(model=MODEL_PATH, weights=WEIGHTS_PATH)
detecter = Predictor(MODEL_DETECTOR)
recognizer = Recognizer(MODEL_OCR)


class Image(BaseModel):
    image_base64: str


app = FastAPI()
security = HTTPBasic()


def check_credential(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


@app.post("/plate_recognition/")
async def upload_file(file: UploadFile = File(...), check_credential: bool = Depends(check_credential)):
    file_name = file.filename
    _, file_format = os.path.splitext(file_name)
    if file_format not in FORMATE_SUPPORT or file_name is None:
        raise HTTPException(status_code=405, detail="Error Format File!")
    try:
        file_path = os.path.join('tmp', file_name)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        image = cv2.imread(file_path)
        image_plates, coordinates = detecter(file_path)
        if image_plates:
            print(image_plates[0].shape)
            labels = recognizer(image_plates)
            image = draw_box(image, coordinates)
            cv2.imwrite("test.png", image)
            os.remove(file_path)
            data = {"image": image_array_to_base64(image)}
            plates = dict()
            for i in range(len(image_plates)):
                plates[labels[i]] = image_array_to_base64(
                    (image_plates[i]).astype(np.uint8))
            data["plates"] = plates
            print(data)
            return {"status": True,
                    "file_name": file_name,
                    "format": file_format,
                    "data": data}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error Core!")

