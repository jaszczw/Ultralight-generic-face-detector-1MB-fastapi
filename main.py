from fastapi import FastAPI, Form ,File , UploadFile
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import os
import MNN
import cv2
import base64
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath('./Ultra-Light-Fast-Generic-Face-Detector-1MB-master/MNN/python'))
import ultraface_py_mnn

app = FastAPI(
    title="Ultra light face detection",
    description="lightweight face detection",
    version='0.0.1',
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=["*"]
    )



class Meta(BaseModel):
    image: str = None
    bbox: List[float] = None
    score : float = None
    text: str = None
    audio: str = None

class DetectionResults(BaseModel):
    image: str = None
#     metas: List[Meta] = None

@app.get("/")
def tester():
    return{"hello worlds"}

@app.post('/predict')
def post_data_v1(
    image_base64: str = Form(...)
    ):
    content = base64.b64decode(image_base64)
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    boxes,label,probs = ultraface_py_mnn.inference(img)
    output = []
    for i in range(len(boxes)):
        box = [int(b) for b in boxes[i]]
        # [x1,y1:x2:y2]
        cropped_img= img[box[1]:box[3],box[0]:box[2]]
        retval, buffer = cv2.imencode('.jpg', cropped_img)
        derived_b64_str = base64.b64encode(buffer)
        # print(cropped_img)
        output.append({
                "bbox": box,
                "score" : probs[i],
                "text": None,
                "audio": None,
                "image" : derived_b64_str
        })

    return output
