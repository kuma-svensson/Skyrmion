import cv2
import numpy as np
import os
import re
import csv
import copy 
from tkinter import filedialog
from typing import Tuple

PATTERN=r'([0-9]*)'
DIR = 'D:\\'
DEFAULT=False
DEBUG=True
SAVE_IMG=True

#Load video
def loadVideo()-> Tuple[str, cv2.VideoCapture]:
    #Load the specified video
    if DEFAULT:
        filename = "E:\\1_20200129_123410.mp4"
    #Select and load videos
    else:
        filename = filedialog.askopenfilename(initialdir = dir)
        
    #Exit if file is invalid
    if not filename:
        exit(0)
    else:
        os.chdir(os.path.dirname(filename))
        cap=cv2.VideoCapture(os.path.basename(filename))
    return filename,cap



#Load background image (for background subtraction method)
def loadBackground(filename)->cv2.typing.MatLike:
    os.chdir("..\\backgrounds")
    try:
        filename, extension = os.path.splitext(filename)
        background=cv2.imread("average_"+os.path.basename(filename)+".png")
        return background
    except Exception:
        exit(0)


def loadROI()->cv2.typing.MatLike:
    try:
        roi=cv2.imread("roi.png")
        return roi
    except Exception:
        exit(0)

#Generate the average image of frames in a specific section of a video
def makeBackgroundImage(name:str,cap:cv2.VideoCapture,start_frame, end_frame)->cv2.typing.MatLike:
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Adjust frame number within range
    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = min(total_frames - 1, max(end_frame, start_frame))
    
    # Initialize average image
    sum_frame = None
    frame_count = 0
    
    # Load a specified range of frames and calculate the average image
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        ret, frame = cap.read()
        if ret:
            if sum_frame is None:
                sum_frame = frame.astype(float)
            else:
                sum_frame += frame
            frame_count += 1
        else:
            break
    # Calculate average image

    if frame_count > 0:
        average_frame = (sum_frame / frame_count).astype(dtype='uint8')
    else:
        average_frame = None
        
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    cv2.imshow("Mean Frame", average_frame)
    cv2.waitKey(0)
    return average_frame
    

class Analysis:
    def __init__(self,video:cv2.VideoCapture,background:cv2.typing.MatLike):
        try:
            self.video=video
            self.fps=video.get(cv2.CAP_PROP_FPS)
            self.ret,self.frame_prev=video.read()
        except Exception:
            exit(0)
        
        self.kernel=np.ones((3, 3), np.uint8)
        self.bubbles=np.zeros((0,3))
        self.M_dict={}
        self.M_dict_old={}
        self.background=background
        print(self.fps,self.ret)
    
    def startAnalysis(self):
        if DEBUG:
            fgbg = cv2.createBackgroundSubtractorMOG2() 
        endframe=int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        frameAddweight=copy.copy(self.frame_prev)
        result=copy.copy(self.background)
        
        for cnt in range(0,endframe-1): 
            ret,frame_next=self.video.read()
            frameAddweight=cv2.addWeighted(frameAddweight,0.5,frame_next,0.5,0)
            frameAddweightGray=cv2.cvtColor(frameAddweight, cv2.COLOR_BGR2GRAY)
            frameSubstracted=cv2.absdiff(background,frameAddweight)
            frameSubstracted = cv2.medianBlur(frameSubstracted,3)
            _,frameBinary=cv2.threshold(frameSubstracted,12,255,cv2.THRESH_BINARY)
            
            frameErosion=cv2.erode(frameBinary,self.kernel,iterations=1)
            frameDilation=cv2.dilate(frameErosion,self.kernel,iterations = 2)
            frameDilationGray=cv2.cvtColor(frameDilation,cv2.COLOR_BGR2GRAY)
            
            
            contours,_=cv2.findContours(frameDilationGray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            self.M_dict={}
            result=copy.copy(result)
            for k in range(len(contours)):
                moments=cv2.moments(contours[k])
                if 0<moments["m00"]<200:
                    self.M_dict[str(k)]=moments
                    pos=[int(moments["m10"] / moments["m00"]),int(moments["m01"] / moments["m00"])]
                    cv2.circle(result,(pos[0],pos[1]),1, thickness=-1,color=[0,0,0])
                    self.bubbles=np.append(self.bubbles,np.array([[cnt,pos[0],pos[1]]]),axis=0)
            
            if SAVE_IMG:
                cv2.imwrite('{}_{}.{}'.format("H:\\learning\\project\\Tracking\\sample_data\\test\\", str(cnt), "png"), frameSubstracted)
            
            if DEBUG:
                concatenated_image = cv2.hconcat([result, frameSubstracted])
                
                cv2.putText(concatenated_image,str(cnt) , (0, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), lineType=cv2.LINE_AA)
                cv2.imshow('comparison', concatenated_image) 
                if cv2.waitKey(int(1))==27:
                    pass
        np.savetxt(".\\result.csv",self.bubbles,delimiter=",",header="frame,x,y")
                
if __name__=='__main__':
    filename,cap=loadVideo()
    background=makeBackgroundImage(filename,cap,800,20015)
    analysis=Analysis(cap,background)
    analysis.startAnalysis()
    
