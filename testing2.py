import cv2
def Life2CodingRGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  # checks mouse moves
        colorsBGR = image[y, x]
        colorsRGB=tuple(reversed(colorsBGR)) #Reversing the OpenCV BGR format to RGB format
        print("RGB Value at ({},{}):{} ".format(x,y,colorsRGB))
# Read an image
image = cv2.imread("woozie.png")
# Create a window and set Mousecallback to a function for that window
cv2.namedWindow('Life2CodingRGB')
cv2.setMouseCallback('Life2CodingRGB', Life2CodingRGB)
# Do until esc pressed
while(True):
    cv2.imshow('Life2CodingRGB', image)
    if cv2.waitKey(10) & 0xFF == 27:
        break
# if esc is pressed, close all windows.
cv2.destroyAllWindows()