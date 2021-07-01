# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

#--------------------------------------#
## Taken from https://www.pyimagesearch.com/2015/09/14/ball-tracking-with-opencv/#download-the-code ##
def ball_detection(frame, upper_range, lower_range, image_width=600):
  ball_x = None
  ball_y = None
  ball_radius = None
  center = None

  # resize the frame, blur it, and convert it to the HSV color space
  frame = imutils.resize(frame, width=image_width)
  blurred = cv2.GaussianBlur(frame, (11, 11), 0)
  hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

  # construct a mask for the color "green", then perform
  # a series of dilations and erosions to remove any small
  # blobs left in the mask
  mask = cv2.inRange(hsv, lower_range, upper_range)
  mask = cv2.erode(mask, None, iterations=2)
  mask = cv2.dilate(mask, None, iterations=2)

  # find contours in the mask and initialize the current
  # (x, y) center of the ball
  cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  center = None

  # only proceed if at least one contour was found
  if len(cnts) > 0:
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    M = cv2.moments(c)
    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    # only proceed if the radius meets a minimum size
    if radius > 10:
      # draw the circle and centroid on the frame,
      # then update the list of tracked points
      cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
      cv2.circle(frame, center, 5, (0, 0, 255), -1)

      ball_x, ball_y, ball_radius, center = int(x), int(y), int(radius), center

  return frame, mask, ball_x, ball_y, ball_radius, center
#---_#


class KivyCamera(Image):
    def __init__(self, capture, fps, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.capture = capture
        Clock.schedule_interval(self.update, 1.0 / fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:

            self.deque_maxlen = 64
            self.sensitivity = 40


            ## Follow Ball ##
            lower_white = np.array([0,0,255-self.sensitivity])
            upper_white = np.array([255,self.sensitivity,255])
            

            frame, mask, ball_x, ball_y, ball_radius, center = ball_detection(frame, upper_white, lower_white)

            try:
                self.pts.appendleft(center)
            except:
                self.pts = deque(maxlen=self.deque_maxlen) 
                self.pts.appendleft(center)

            ## loop over the set of tracked points
            for i in range(1, len(self.pts)):
                # if either of the tracked points are None, ignore
                # them
                if self.pts[i - 1] is None or self.pts[i] is None:
                    continue
                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(self.deque_maxlen / float(i + 1)) * 2.5)
                cv2.line(frame, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

            ## convert it to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            image_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            
            ## display image from the texture
            self.texture = image_texture


class CamApp(App):

    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.my_camera = KivyCamera(capture=self.capture, fps=30)

        
        return self.my_camera

    def on_stop(self):
        #without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == '__main__':
    CamApp().run()