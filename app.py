from flask import Flask, flash, request, redirect, url_for
from flask_cors import CORS, cross_origin
from model import predict
import cv2 as cv
import numpy as np
# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = ''


@app.route('/', methods=['POST', 'GET'])
@cross_origin(origin='*')
def home():
    if request.method == 'POST':
        image = request.files['file'].read()
        file_bytes = np.fromstring(image, np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)  
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        res = predict(img)
        return res
    if request.method == 'GET':
        return 'get'
    return ''


# Start Backend
if __name__ == '__main__':
    app.run()

