# Run this script in the face_trip conda environment
import cv2
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
from ultralytics import YOLO
import uuid
import pandas as pd
"""
The model detects the following emotions
(Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, or Surprise) 
Can check the emotions and check if negative emotions are influencing the driving behaviour
"""



column_name=['Timestamp','emotion']
df=pd.DataFrame(columns=column_name)

model_name='enet_b0_8_best_afew'
device='cuda' if torch.cuda.is_available() else 'cpu'
model_yolo=YOLO(r'E:\tensorrt\Trip\models\yolov8n-face.pt')
# model_name='models\enet_b2_8_best.pt'
fer=HSEmotionRecognizer(model_name=model_name,device=device) 

path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'
cam=cv2.VideoCapture(path)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
fps=int(cam.get(cv2.CAP_PROP_FPS))

fileid=str(uuid.uuid4())
record_path=f'emotion_recognition_{fileid}.avi'
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2
video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

while cam.isOpened():
    ret,frame=cam.read()
    if not ret:
        print("End of Video")
        break

    frame_head=frame[height//2:,:width//2]  #Snipping the bottom left face frame
    timestamp = int(cam.get(cv2.CAP_PROP_POS_MSEC))   
    result=model_yolo(frame_head)
    boxes=result[0].boxes
    for box in boxes:
        det_list=box.xyxy.tolist()[0]
        top_x=int(det_list[0])
        top_y=int(det_list[1])
        bottom_x=int(det_list[2])
        bottom_y=int(det_list[3])
        frame_face=frame_head[top_y:bottom_y+1,top_x:bottom_x+1].copy()
        cv2.rectangle(frame_head,(top_x,top_y),(bottom_x,bottom_y),(255,0,0),2)
        emotion,scores=fer.predict_emotions(frame_face,logits=True)
        cv2.putText(frame_head,str(emotion), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        new_value= pd.Series([timestamp,str(emotion)], index=column_name)
        df.loc[len(df)] = new_value
        

    # frame_head=cv2.resize(frame_head,(640,480))  
    video_write.write(frame_head)
    cv2.imshow('cam_bottom_left',frame_head)
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break

df.to_csv('emotion_detect.csv',index=False)
cam.release()
video_write.release()
cv2.destroyAllWindows()