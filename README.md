# Ultralight-face-detect-fastapi
deploying of ultralight general face detection model to an API using fastapi

The main script for the api deployment is "main.py" . Simply start the API server with "uvicorn main:app --reload" and test the model in the api server
Input takes in a base64 string and the model does the detection and outputs bbox,confidence score and cropped base64 str .

You can encode the image here https://www.base64-image.de/ and decode the output image here https://codebeautify.org/base64-to-image-converter to see the cropped face

Implemented from https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

credits to the model used in the git repo above
