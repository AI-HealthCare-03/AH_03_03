# -*- encoding: utf-8 -*-
# This file supports web-based object classfication
# by sangkny
# -------------------------------------------------
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import cv2
import numpy as np

from food_classifier_yolo import food_classifier_Json
# classification function

app = Flask(__name__)
#App name

def recognize(filename):
    image = cv2.imread(filename) # be careful for hangul name
    # read file and put it into an image array
    return food_classifier_Json(image=image)
# classification for food images

import base64

# for hangul file name
def recognizeBase64(base64_code):
    file_bytes = np.asarray(bytearray(base64.b64decode(base64_code)),dtype=np.uint8)
    image_data_ndarray = cv2.imdecode(file_bytes,1)
    return food_classifier_Json(image_data_ndarray)

import time

@app.route('/uploader', methods=['GET', 'POST'])# request routing
def upload_file():
    if request.method == 'POST':
        # if POST case
        f = request.files['file']
        f.save("./images_rec/"+secure_filename(f.filename))
        # saving the requested file
        t0 = time.time()
        res = recognize("./images_rec/"+secure_filename(f.filename))
        print("elapsed time:",time.time() - t0)
        return res
        # return the result

        # return 'file uploaded successfully'
    return render_template('upload.html')

if __name__ == '__main__':
    # input
    ip_address = "0.0.0.0"#"127.0.0.1"
    port_number = 3000 #8000
    app.run(ip_address,port=int(port_number))
    # run app with ip and port numbers

