# Ultralight-genric-face-detector-1MB-fastapi
deploying of ultralight general face detection model to an API using fastapi

The main script for the api deployment is "main.py" . Simply start the API server with "uvicorn main:app --reload" and test the model in the api server

Input takes in a base64 string and the model does the detection and outputs bbox,confidence score and cropped base64 str .

You can encode the image here https://www.base64-image.de/ and decode the output image here https://codebeautify.org/base64-to-image-converter to see the cropped face

**Example input**
![2](https://user-images.githubusercontent.com/71302213/115052802-f3973c00-9f10-11eb-85f3-1689c9ed08f9.jpg)


**Running the API**
![Capture](https://user-images.githubusercontent.com/71302213/115053224-7ae4af80-9f11-11eb-9434-b6d20aad67cc.PNG)

**decode the output string**
![Capture](https://user-images.githubusercontent.com/71302213/115053391-a9628a80-9f11-11eb-84f3-918dfb871527.PNG)





Implemented from https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB

credits to the model used in the git repo above

