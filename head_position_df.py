import cv2
import uuid
from ultralytics import YOLO
from sixdrepnet import SixDRepNet
import numpy as np
import pandas as pd

column_name=['Timestamp','Pitch','Yaw','Roll']
df=pd.DataFrame(columns=column_name)

model = SixDRepNet()
model_yolo=YOLO(r'models\yolov8n-face.pt')
path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'

fileid=str(uuid.uuid4())
record_path=f'head_tilt_{fileid}.avi'
fourcc=cv2.VideoWriter_fourcc(*'mp4v')

cap= cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))

resize_width=width//2
resize_height=height//2
video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

while cap.isOpened():
    # print("Entered loop")
    ret,frame=cap.read()
    if not ret:
        print("End of Video")
        break
    frame_head=frame[height//2:,:width//2]  #Snipping the bottom left face frame
    timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))   
    result=model_yolo(frame_head)
    boxes=result[0].boxes
    for box in boxes:
        det_list=box.xyxy.tolist()[0]
        top_x=int(det_list[0])
        top_y=int(det_list[1])
        bottom_x=int(det_list[2])
        bottom_y=int(det_list[3])
        cv2.rectangle(frame_head,(top_x,top_y),(bottom_x,bottom_y),(255,0,0),2)
        # frame_face=frame[top_y:bottom_y+1,top_x:bottom_x+1].copy()
        frame_face=frame_head[top_y:bottom_y+1,top_x:bottom_x+1]
        pitch, yaw, roll = model.predict(frame_face)
        
        pitch_round=round(float(pitch[0]),3)
        yaw_round=round(float(yaw[0]),3)
        roll_round=round(float(roll[0]),3)
        # print(type(pitch_round),"-----------------------------------------------")
        # Appending the row values into dataframe
        # new_values_df = pd.DataFrame([timestamp,pitch_round,yaw_round,roll_round])
        
        new_value= pd.Series([timestamp,pitch_round,yaw_round,roll_round], index=column_name)
        df.loc[len(df)] = new_value
        

        pitch_string=f'Pitch: {pitch_round}'
        yaw_string=f'Yaw: {yaw_round}'
        roll_string=f'Roll: {roll_round}'
        cv2.putText(frame_head,pitch_string, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_head,yaw_string, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_head,roll_string, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        model.draw_axis(frame_face, yaw, pitch, roll)
        print(pitch_round, yaw_round, roll_round)

    # frame_head=cv2.resize(frame_head,(640,480))  
    video_write.write(frame_head)
    
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty('test', cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow('test', 640, 480)
    cv2.imshow('test',frame_head)
    # cv2.setWindowProperty('test', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # cv2.setWindowProperty('test', cv2.WND_PROP_TOPMOST, 1)
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break

df.to_csv('head_roll.csv',index=False)
cap.release()
video_write.release()
cv2.destroyAllWindows()