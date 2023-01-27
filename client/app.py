import sys
sys.path.append('../')

from lib.utils.image_transforms import scale_crop
import requests
import imageio
import cv2
import base64
from flask import Flask, render_template, Response, redirect, url_for, request
import numpy as np
from lib.utils.profiling import frame_count, timer

import threading

from videostream import camera_frame_lock, camera_frame, get_next_frame, get_frame
from lib.face_detection import get_facial_roi, get_face_location, get_source_frame
import pickle

app = Flask(__name__)
# camera = VideoStream(src=0).start() #cv2.VideoCapture(0)

server_url = 'http://127.0.0.1:5000'
# server_url = 'http://9aaa3ade2048.ngrok.io'
# server_url = "http://34.221.75.41:5000"
configure_url = server_url + "/configure"
transform_url = server_url + "/transform"

server_state = None
NULL_UID = '#'
jpeg_quality = 95

# source_url = 'static/images/jack3.jpg'
# source_image = imageio.imread(source_url)
# source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
# source_image = cv2.flip(scale_crop(source_image), 1)
transform_pos = (0,0)
transform_size = (256,256)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame 
        frame = get_frame()

        if frame is None:
            continue
        else:
            frame = cv2.flip(scale_crop(frame), 1)
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            stream_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + stream_bytes + b'\r\n')


@app.route('/configure', methods=['POST'])
def configure():

    frame = get_frame()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.flip(scale_crop(frame), 1)
    _, frame_buffer = cv2.imencode('.jpg', frame)

    frame_encoded = base64.b64encode(frame_buffer)
    config = {}
    if request.method == "POST":
        if request.form and 'config' in request.form:
            config = eval(request.form['config'])
    config_pickle = base64.b64encode(pickle.dumps(config))
    data = {
        "config": config_pickle,
        "frame": frame_encoded,
    }

    r = requests.post(configure_url, data=data)

    if r.status_code != 200:
        print("error")
        return ('', 204)

    global server_state
    server_state = r.text

    return redirect(url_for('index'))


def gen_transformed_frames():
    while True:
        # Capture frame-by-frame
        frame = get_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame is None:
            break
        else:
            frame = cv2.flip(scale_crop(frame), 1)
            _, frame_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            frame_encoded = base64.b64encode(frame_buffer)
            data = {
                "config": "",
                "frame": frame_encoded
            }
            # frame_count()
            # timer('')
            r = requests.post(transform_url, data=data)
            # print(timer('server_processing'))
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + r.content + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/transform')
def transform():
    if server_state and server_state is not NULL_UID:
        return Response(gen_transformed_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/')
def index():
    """Video streaming home page."""
    if server_state and server_state is not NULL_UID :
        return render_template('transformed.html', transform_pos=transform_pos, transform_size=transform_size, server_state=server_state)
    return render_template('index.html')


if __name__ == '__main__':
    t = threading.Thread(target=get_next_frame, args=(24,))
    t.daemon = True
    t.start()
    app.run(port=3000)
