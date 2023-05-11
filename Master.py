
import cv2
import numpy as np
import math
import Jetson.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)  
#GPIO.setup(#of pin,GPIO.OUT)
GPIO.setwarnings(False)
 
#FL:
FL_IN1=7

FL_IN2=11

#FR:
FR_IN1=12
FR_IN2=15

#BF:
BL_IN1=19
BL_IN2=21

#BR:
BR_IN1=26
BR_IN2=23

ENA1=31
ENA2=32

#SL
SL_INT1=38  #in1_YELLOW
SL_INT2=40  #IN2_BLUE

#CR
CR_IN3=13
CR_IN4=22

switch_top=35
switch_bot=24

GPIO.setup(CR_IN3,GPIO.OUT)
GPIO.setup(CR_IN4,GPIO.OUT)

GPIO.setup(switch_top,GPIO.OUT)
GPIO.setup(switch_bot,GPIO.OUT)
GPIO.output(switch_top, GPIO.HIGH)
GPIO.output(switch_bot, GPIO.HIGH)

#Initialize FL_ENA, FL_IN1, and FL_IN2
#FL_PWM= [pin_num] #ENA
GPIO.setup(FL_IN1,GPIO.OUT)
GPIO.setup(FL_IN2,GPIO.OUT)

#Initialize FR_ENA, FR_IN1, and FR_IN2
#FR_PWM= [pin_num] #ENA
GPIO.setup(FR_IN1,GPIO.OUT)
GPIO.setup(FR_IN2,GPIO.OUT)

#Initialize BL_ENA, BL_IN1, and BL_IN2
#BL_PWM= [pin_num] #ENA
GPIO.setup(BL_IN1,GPIO.OUT)
GPIO.setup(BL_IN2,GPIO.OUT)

#Initialize BR_ENA, BR_IN1, and BR_IN2
#BR_PWM= [pin_num] #ENA
GPIO.setup(BR_IN1,GPIO.OUT)
GPIO.setup(BR_IN2,GPIO.OUT)

GPIO.setup(SL_INT1,GPIO.OUT)
GPIO.setup(SL_INT2,GPIO.OUT)

GPIO.setup(ENA1,GPIO.OUT)
GPIO.setup(ENA2,GPIO.OUT)
GPIO.output(ENA1, GPIO.HIGH)
GPIO.output(ENA2, GPIO.HIGH)

def backward():
    print('backward')
    # FL_Forward
    GPIO.output(FL_IN1, GPIO.HIGH)
    GPIO.output(FL_IN2, GPIO.LOW)

    # FR_Forward
    GPIO.output(FR_IN1, GPIO.HIGH)
    GPIO.output(FR_IN2, GPIO.LOW)

    # BL_Forward
    GPIO.output(BL_IN1, GPIO.HIGH)
    GPIO.output(BL_IN2, GPIO.LOW)

    # BR_Forward
    GPIO.output(BR_IN1, GPIO.HIGH)
    GPIO.output(BR_IN2, GPIO.LOW)
    #x='z'

        
def forward():
    print('forward')
    # FL_Backward
    GPIO.output(FL_IN1, GPIO.LOW)
    GPIO.output(FL_IN2, GPIO.HIGH)
    
    # FR_Backward
    GPIO.output(FR_IN1, GPIO.LOW)
    GPIO.output(FR_IN2, GPIO.HIGH)

    # BL_Backward
    GPIO.output(BL_IN1, GPIO.LOW)
    GPIO.output(BL_IN2, GPIO.HIGH)

    # BR_Backward
    GPIO.output(BR_IN1, GPIO.LOW)
    GPIO.output(BR_IN2, GPIO.HIGH)
    #x='z'   
        
def stop():
    print('stop')
    # FL_Stop
    GPIO.output(FL_IN1, GPIO.LOW)
    GPIO.output(FL_IN2, GPIO.LOW)

    # FR_Stop
    GPIO.output(FR_IN1, GPIO.LOW)
    GPIO.output(FR_IN2, GPIO.LOW)

    # BL_Stop
    GPIO.output(BL_IN1, GPIO.LOW)
    GPIO.output(BL_IN2, GPIO.LOW)

    # BR_Stop
    GPIO.output(BR_IN1, GPIO.LOW)
    GPIO.output(BR_IN2, GPIO.LOW)
    #x='z'
        
def ccw():
    print('ccw')
    # FL_Left
    GPIO.output(FR_IN1, GPIO.LOW)
    GPIO.output(FR_IN2, GPIO.HIGH)

    # FR_Left
    GPIO.output(FL_IN1, GPIO.HIGH)
    GPIO.output(FL_IN2, GPIO.LOW)

    # BL_Left
    GPIO.output(BR_IN1, GPIO.HIGH)
    GPIO.output(BR_IN2, GPIO.LOW)

    # BR_Left
    GPIO.output(BL_IN1, GPIO.LOW)
    GPIO.output(BL_IN2, GPIO.HIGH)
    
def cw():
    print('cw')
    # FL_Right
    GPIO.output(FR_IN1, GPIO.HIGH)
    GPIO.output(FR_IN2, GPIO.LOW)

    # FR_Right
    GPIO.output(FL_IN1, GPIO.LOW)
    GPIO.output(FL_IN2, GPIO.HIGH)

    # BL_Right
    GPIO.output(BR_IN1, GPIO.LOW)
    GPIO.output(BR_IN2, GPIO.HIGH)

    # BR_Right
    GPIO.output(BL_IN1, GPIO.HIGH)
    GPIO.output(BL_IN2, GPIO.LOW)
        
def right():
    print('right')
    # FL_CCW
    GPIO.output(FL_IN1, GPIO.LOW)
    GPIO.output(FL_IN2, GPIO.HIGH)

    # FR_CCW
    GPIO.output(FR_IN1, GPIO.HIGH)
    GPIO.output(FR_IN2, GPIO.LOW)

    # BL_CCW
    GPIO.output(BL_IN1, GPIO.LOW)
    GPIO.output(BL_IN2, GPIO.HIGH)

    # BR_CCW
    GPIO.output(BR_IN1, GPIO.HIGH)
    GPIO.output(BR_IN2, GPIO.LOW)
        
def left():
    print('left')
    # FL_CCW
    GPIO.output(FL_IN1, GPIO.HIGH)
    GPIO.output(FL_IN2, GPIO.LOW)

    # FR_CCW
    GPIO.output(FR_IN1, GPIO.LOW)
    GPIO.output(FR_IN2, GPIO.HIGH)

    # BL_CCW
    GPIO.output(BL_IN1, GPIO.HIGH)
    GPIO.output(BL_IN2, GPIO.LOW)

    # BR_CCW
    GPIO.output(BR_IN1, GPIO.LOW)
    GPIO.output(BR_IN2, GPIO.HIGH)
        
def goUp():
    GPIO.output(SL_INT1,GPIO.HIGH)
    GPIO.output(SL_INT2,GPIO.LOW)

def goDown():
    GPIO.output(SL_INT1,GPIO.LOW)
    GPIO.output(SL_INT2,GPIO.HIGH)

def SLstop():
    GPIO.output(SL_INT1,GPIO.LOW)
    GPIO.output(SL_INT2,GPIO.LOW)

def SLrun():
    start_up=time.time()
    end_up=start_up+16     # 16 seconds
    while(time.time()<=end_up):
        goUp()
    SLstop()

    print("charging time")
    charging_time=5
    #time.sleep(charging_time) # 5 secs of changing time
    for i in range(charging_time,1,-1):
        print("charging time: "+ str(i))
        cv2.putText(frame,("charging time: "+str(i)),(100,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        time.sleep(1)

    start_down=time.time()
    end_down=start_down+16
    while(time.time()<=end_down):
        goDown()
    SLstop()
    GPIO.cleanup()

def long():
    print("run")
    GPIO.output(CR_IN3, GPIO.HIGH)
    GPIO.output(CR_IN4, GPIO.LOW)
def short():
    print("run")
    GPIO.output(CR_IN3, GPIO.HIGH)
    GPIO.output(CR_IN4, GPIO.LOW)
def CRstop():
    GPIO.output(CR_IN3, GPIO.LOW)
    GPIO.output(CR_IN4, GPIO.LOW)
def CR_forward():
    while True:
        if(switch_top==0):
            print("cable is too tense")
            long()
        elif(switch_bot==0):
            print("cable is too loose")
            short()
        else:
            print("releasing")
            long()
def CR_backward():
    while True:
        if(switch_top==0):
            print("cable is too tense")
            long()
        elif(switch_bot==0):
            print("cable is too loose")
            short()
        else:
            print("unwinding")
            short()


def goBack():
    #backward()
    cap=cv2.VideoCapture(0)
    while flag:
    # read the frame from the camera
        ret, frame = cap.read()
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lower_tape = np.array([96,105,0]) # [0, 0, 0]
        upper_tape = np.array([120,255,255]) # [50, 50, 50]

        lower_home=np.array([108,0,0])
        upper_home=np.array([179,235,255])

        mask = cv2.inRange(frame, lower_tape, upper_tape)
        mask_blue=cv2.inRange(hsv,lower_tape,upper_tape)
        mask_home=cv2.inRange(hsv,lower_home,upper_home)

        kernel=np.ones((3,3),np.uint8)
        mask=cv2.erode(mask_blue,kernel,iterations=1)
        mask=cv2.dilate(mask_blue,kernel,iterations=5)
        '''if cv2.countNonZero(mask) == 0:
                cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)'''
        if cv2.countNonZero(mask_blue)==0:
            cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)
            stop()

        contours, hierarchy=cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #cv2.drawContours(frame,contours,-1,(0,255,0),3) 

        if len(contours)>0:
            x,y,w,h=cv2.boundingRect(contours[0])
            #finding minimum area rotated, returning (center(x,y),(w,h),angle of rotation)
            min_box=cv2.minAreaRect(contours[0])
            
            (x_min,y_min),(w_min,h_min),angle=min_box
            box=cv2.boxPoints(min_box)
            box=np.intp(box)
            angle=int(angle)
            

            frame_center_x = frame.shape[1] // 2 
            frame_center_y = frame.shape[0] // 2
            #cv2.circle(frame,(frame_center_x,frame_center_y),5,(255,0,0),-1) #mid of screen, blue
            cv2.circle(frame,(frame_center_x-90,frame_center_y),5,(255,0,0),-1) #mid left of screen, blue
            cv2.circle(frame,(frame_center_x+90,frame_center_y),5,(255,0,0),-1) #mid right of screen, blue
            cv2.circle(frame,(int(x_min),int(y_min)),5,(0,255,0),-1) # mid of bounding box, green

            if angle<45: 
                angle=90-angle
                
            else:
                angle=180-angle
            
            cv2.drawContours(frame,[box],-1,(0,0,255),3)
            cv2.drawContours(mask_blue,[box],-1,(0,0,255),3)
            cv2.putText(frame,("angle="+str(angle)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            
            if angle>=80 and angle<=100:
                if x_min<frame_center_x-140:  #90orig
                    print("slide left")
                    cv2.putText(frame,("slide right"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    left()
                elif x_min>frame_center_x+140: #90
                    print("slide right")
                    cv2.putText(frame,("slide left"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    right()
                else:   
                    cv2.putText(frame,("go back"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                    backward()
            elif angle<80:
                cv2.putText(frame,("turn right"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                cw()
            elif angle>100:
                cv2.putText(frame,("turn left"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                ccw()
            else:
                cv2.putText(frame,("stop"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                stop()

        cv2.resize(frame,(640,480))
        cv2.imshow("frame",frame)
        #cv2.resize(mask,(640,480))
        #cv2.imshow("mask",mask)

        cv2.resize(hsv,(640,480))
        cv2.imshow("mask",mask_blue)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# capture the video from camera
cap = cv2.VideoCapture(0)

flag=True
while flag:
    # read the frame from the camera
    ret, frame = cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    lower_tape = np.array([96,65,0]) # [0, 0, 0]
    upper_tape = np.array([134,255,255]) # [50, 50, 50]

    lower_home=np.array([26,0,0])
    upper_home=np.array([107,235,255])

    mask = cv2.inRange(frame, lower_tape, upper_tape)
    mask_blue=cv2.inRange(hsv,lower_tape,upper_tape)
    mask_home=cv2.inRange(hsv,lower_home,upper_home)

    kernel=np.ones((3,3),np.uint8)
    mask=cv2.erode(mask_blue,kernel,iterations=1)
    mask=cv2.dilate(mask_blue,kernel,iterations=5)
    '''if cv2.countNonZero(mask) == 0:
         cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)'''
    if cv2.countNonZero(mask_blue)==0:
        cv2.putText(frame,("stop"),(10,400),cv2.FONT_HERSHEY_SIMPLEX,1,(200,100,255),2)
        stop()
        time.sleep(10)
        SLrun()
        goBack()
    contours, hierarchy=cv2.findContours(mask_blue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame,contours,-1,(0,255,0),3) 

    if len(contours)>0:
        x,y,w,h=cv2.boundingRect(contours[0])
        #finding minimum area rotated, returning (center(x,y),(w,h),angle of rotation)
        min_box=cv2.minAreaRect(contours[0])
        
        (x_min,y_min),(w_min,h_min),angle=min_box
        box=cv2.boxPoints(min_box)
        box=np.intp(box)
        angle=int(angle)
        

        frame_center_x = frame.shape[1] // 2 
        frame_center_y = frame.shape[0] // 2
        #cv2.circle(frame,(frame_center_x,frame_center_y),5,(255,0,0),-1) #mid of screen, blue
        cv2.circle(frame,(frame_center_x-90,frame_center_y),5,(255,0,0),-1) #mid left of screen, blue
        cv2.circle(frame,(frame_center_x+90,frame_center_y),5,(255,0,0),-1) #mid right of screen, blue
        cv2.circle(frame,(int(x_min),int(y_min)),5,(0,255,0),-1) # mid of bounding box, green

# put this in a while loop to break when it goes straight 
  
        '''if x_min<(frame_center_x-50):
            cv2.putText(frame,("straft right"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        elif x_min>(frame_center_x+50):
            cv2.putText(frame,("straft left"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
        else:
            cv2.putText(frame,("in range"),(300,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)'''
 
        if angle<45: 
            angle=90-angle
            
        else:
            angle=180-angle
        
        cv2.drawContours(frame,[box],-1,(0,0,255),3)
        cv2.drawContours(mask_blue,[box],-1,(0,0,255),3)
        cv2.putText(frame,("angle="+str(angle)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        '''white_pix=cv2.countNonZero(mask_home)
        black_pix=mask_home.size-white_pix
        if white_pix>black_pix:
            print("stop")
            cv2.putText(frame,("stop motions, start SL"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            stop()
            #SLrun()
            flag=False'''  

        if angle>=80 and angle<=100:
            if x_min<frame_center_x-140: #90origional
                print("slide left")
                cv2.putText(frame,("slide left"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                left()
            elif x_min>frame_center_x+140: #90origional
                print("slide right")
                cv2.putText(frame,("slide right"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                right()
            else:   
                cv2.putText(frame,("go straight"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                forward()
        elif angle<80:
            cv2.putText(frame,("turn right"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cw()
        elif angle>100:
            cv2.putText(frame,("turn left"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            ccw()
        else:
            cv2.putText(frame,("stop"),(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            stop()

    cv2.resize(frame,(640,480))
    cv2.imshow("frame",frame)
    #cv2.resize(mask,(640,480))
    #cv2.imshow("mask",mask)

    cv2.resize(hsv,(640,480))
    cv2.imshow("mask",mask_blue)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close the windows
cap.release()
cv2.destroyAllWindows()

goBack()
cap.release()
cv2.destroyAllWindows()
