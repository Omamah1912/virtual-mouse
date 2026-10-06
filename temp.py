import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False

pyautogui.PAUSE = 0.001

#opening the camera
camera=cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

control_left = 100
control_right = 540
control_top = 50
control_bottom = 430
screen_width,screen_height=pyautogui.size()
previous_x = screen_width / 2
previous_y = screen_height / 2

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
op=vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector=vision.HandLandmarker.create_from_options(op)
timestamp=0
pinch=False
while(1):
    success,frame=camera.read()
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #still an array
    image=mp.Image(image_format=mp.ImageFormat.SRGB, data=frame1) #array converted to a mediapipe image
    results=detector.detect_for_video(image,timestamp) #detect with the image and timestamp
    timestamp+=33
    hand=(results.hand_landmarks)
    if(len(hand)>0):
        index=hand[0][8] #index finger tip
        print(index)
        pixel_x=index.x*640
        pixel_y=index.y*480
        screen_x= np.interp(pixel_x,[control_left,control_right],[0,screen_width])
        screen_y= np.interp(pixel_y,[control_top,control_bottom],[0,screen_height])
        smooothX= previous_x + (screen_x - previous_x) * 0.5
        smooothY= previous_y + (screen_y - previous_y) * 0.5
        pyautogui.moveTo(smooothX,smooothY)
        previous_x = smooothX
        previous_y = smooothY
        index = hand[0][8]
        thumb=hand[0][4]
        #left click
        val=np.linalg.norm(np.array([index.x,index.y])-np.array([thumb.x,thumb.y]))
        if pinch == False:
                pyautogui.click()
                pinch = True
        else:
            pinch = False
        # right click
        middle=hand[0][12]
        val2=np.linalg.norm(np.array([index.x,index.y])-np.array([middle.x,middle.y]))
        if val2 < 0.05:
            pyautogui.click(button='right')
            time.sleep(0.2)

        #scroll
        val3=np.linalg.norm(np.array([index.x,index.y])-np.array([middle.x,middle.y]))

        cv2.circle(frame,(int(pixel_x),int(pixel_y)),10,(0,255,255),cv2.FILLED)

        cv2.imshow("Virtual Mouse",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()