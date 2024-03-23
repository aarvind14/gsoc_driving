from spiga.demo.visualize.plotter import Plotter
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework
import cv2
from ultralytics import YOLO
import numpy as np
import uuid 
import pandas as pd
from functions_utils import eye_aspect_ratio_SPIGA

column_name=['Timestamp','EAR_left','EAR_right','IRIS_left_x','IRIS_left_y','IRIS_right_x','IRIS_right_y']
df=pd.DataFrame(columns=column_name)

dataset='wflw'
processor=SPIGAFramework(ModelConfig(dataset))
plotter=Plotter()

model_path=r'models\yolov8n-face.pt'
model_yolo=YOLO(model_path)

path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'
cam=cv2.VideoCapture(path)

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cam.get(cv2.CAP_PROP_FPS))

fileid=str(uuid.uuid4())
record_path=f'facial_landmarks_SPIGA_{fileid}.avi'

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2

video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2
while cam.isOpened():
    ret,frame=cam.read()
    if not ret:
        print("End of Video")
        break

    cam_bottom_left=frame[height//2:,:width//2]
    result=model_yolo(cam_bottom_left)
    timestamp = int(cam.get(cv2.CAP_PROP_POS_MSEC))  
    boxes=result[0].boxes
    for box in boxes:
        det_list=box.xyxy.tolist()[0]
        top_x=int(det_list[0])
        top_y=int(det_list[1])
        bottom_x=int(det_list[2])
        bottom_y=int(det_list[3])
        cv2.rectangle(cam_bottom_left,(top_x,top_y),(bottom_x,bottom_y),(255,0,0),2)

        w=abs(bottom_x-top_x)
        h=abs(bottom_y-top_y)
        bbox=[top_x,top_y,w,h]
        features=processor.inference(cam_bottom_left,[bbox])
        # Prepare variables
        landmarks=np.array(features['landmarks'][0])
        headpose=np.array(features['headpose'][0])
        x1=round(headpose[0],3)
        x2=round(headpose[1],3)
        x3=round(headpose[2],3)
       
        # # print(headpose)
        cam_bottom_left=plotter.landmarks.draw_landmarks(cam_bottom_left,landmarks,thick=1)
        cam_bottom_left=plotter.hpose.draw_headpose(cam_bottom_left,[top_x,top_y,top_x+w,top_y+h],headpose[:3],headpose[3:],euler=True)
        landmarks_int=[landmark.astype('int') for landmark in landmarks]
        EAR_left,EAR_right=eye_aspect_ratio_SPIGA(landmarks_int=landmarks_int)
        iris_left_x,iris_left_y=landmarks_int[96]
        iris_right_x,iris_right_y=landmarks_int[97]

        new_value= pd.Series([timestamp,EAR_left,EAR_right,iris_left_x,iris_left_y,iris_right_x,iris_right_y], index=column_name)
        EAR_left_round=round(float(EAR_left),3)
        EAR_right_round=round(float(EAR_right),3)
        df.loc[len(df)] = new_value

        text_EAR_left=str(f'EAR_left: {EAR_left_round}')
        text_EAR_right=str(f'EAR_right: {EAR_right_round}')
        
        cv2.putText(cam_bottom_left,text_EAR_left,(10,30),font,font_scale,font_color,thickness)
        cv2.putText(cam_bottom_left,text_EAR_right,(10,70),font,font_scale,font_color,thickness)
        
        
    cv2.imshow('color',cam_bottom_left)
    video_write.write(cam_bottom_left)
    #cv2.imshow('color_crop',color_crop)

    if cv2.waitKey(1) & 0xff == 27:
        print("Quitting the program")
        break

df.to_csv('eye_features.csv',index=False)
cam.release()
video_write.release()
cv2.destroyAllWindows()

