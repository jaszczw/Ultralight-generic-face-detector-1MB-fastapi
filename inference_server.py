import base64

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from inference import Detector
from pathlib import Path
from tempfile import NamedTemporaryFile
import uvicorn
import shutil
import os
import numpy as np
import cv2
import json
import torch
import argparse

# class Meta(BaseModel):
#     cropped: str
#     bbox: List[float]
#     label: str

# class DetectionResults(BaseModel):
#     image: str = None
#     metas: List[Meta] = None


class Meta(BaseModel):
    image: str = None
    bbox: List[float] = None
    text: str = None
    audio: str = None

class DetectionResults(BaseModel):
    # image: str = None
    metas: List[Meta] = None

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:0')

app = FastAPI(
    title="AI API SERVER",
    description="AI api server for training and deployment dispatching",
    version="0.1.1",
)


# allow cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

model = None


@app.get("/test_conn")
def test_connection():
    return {"detail": "connected"}


# @app.post("/inference_det")
# async def detection_inference(file: UploadFile = File(...)):
#     content = await file.read()
#     global model
#     if model != None:
#         nparr = np.fromstring(content, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         res = model.inference(img)
#         return JSONResponse(content=res)
#     else:
#         return {"Should initialize the model"}


@app.post("/inference_det")
async def detection_inference(image_base64: str = Form(...)):
    content = base64.b64decode(image_base64)
    global model
    if model != None:
        nparr = np.fromstring(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        res = model.inference(img)
        retval, buffer = cv2.imencode('.jpg', res)
        res = base64.b64encode(buffer)
        return res
    else:
        return {"Should initialize the model"}

@app.post("/inference_det_detailed", response_model=List[Meta])
async def detection_inference(
        image_base64: str = Form(...),
        object_lists: str = Form(...)
    ):
    content = base64.b64decode(image_base64)
    global model
    if model != None:
        nparr = np.fromstring(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        output = model.detailed_inference(img, object_lists)
        # retval, buffer = cv2.imencode('.jpg', img)
        # output["image"] = base64.b64encode(buffer)
        return output

        # for seg in res["cropped"]:
        #     retval, buffer = cv2.imencode('.jpg', res)
        #     temp.append(base64.b64encode(buffer))
        # res["cropped"] = temp
        # return res
    else:
        return {"Should initialize the model"}


if __name__ == '__main__':
    args = parser.parse_args()

    # cuda = 'cpu'

    torch.cuda.init()
    cuda = args.cuda

    # global model
    config = os.path.join('./detection', 'configs', 'faster_configs.py')
    checkpoint = os.path.join('./detection', 'checkpoint', 'checkpoint.pth')
    model = Detector(config, checkpoint, cuda)

    uvicorn.run(app=app, host='0.0.0.0', port=5111, debug=False)
    # uvicorn.run(app=app, host='127.0.0.1', port=5111, debug=False)
