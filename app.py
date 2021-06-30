from flask import Flask, render_template, Response
import cv2
import numpy as np
import detection
from collections import deque
import time

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:

            '''
            sensitivity = 40
            lower_white = np.array([0,0,255-sensitivity])
            upper_white = np.array([255,sensitivity,255])
            pts = deque(maxlen=64)
            vs = cv2.VideoCapture(0)
            # allow the camera or video file to warm up
            time.sleep(2.0)

            frame, mask, ball_x, ball_y, ball_radius, center = detection.ball_detection(frame, upper_white, lower_white)

            pts.appendleft(center)

            # loop over the set of tracked points
            for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

            '''
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)