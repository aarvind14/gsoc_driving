import cv2
from l2cs import Pipeline, render
import torch
import pandas as pd
import uuid
import numpy as np

column_name=['Timestamp','count','pitch_eye','yaw_eye']
df=pd.DataFrame(columns=column_name)

gaze_pipeline = Pipeline(
    weights='models\L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu'), # or 'gpu',
    confidence_threshold=0.8
)
 
path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'
fileid=uuid.uuid4()
cam=cv2.VideoCapture(path)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
fps=int(cam.get(cv2.CAP_PROP_FPS))

record_path=f'gaze_test_{fileid}.avi'

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2

video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255, 255, 255)
thickness = 2
count=1

while cam.isOpened():
    try:
        ret,frame=cam.read()
        if not ret:
            print("End of Video")
            break
        cam_bottom=frame[height//2:,:width//2]
        results = gaze_pipeline.step(cam_bottom)
        timestamp = int(cam.get(cv2.CAP_PROP_POS_MSEC))  
        frame_test = render(cam_bottom, results)
        pitch,yaw=results.pitch[0],results.yaw[0]
        row_val=[timestamp,count,pitch,yaw]
        df.loc[len(df)] = row_val
        text_pitch='Pitch: '+str(np.round(results.pitch[0], decimals=5))
        text_yaw='Yaw: '+str(np.round(results.yaw[0], decimals=5))
        cv2.putText(frame_test,text_pitch,(10,30),font,font_scale,font_color,thickness)
        cv2.putText(frame_test,text_yaw,(10,70),font,font_scale,font_color,thickness)
        cv2.imshow('gaze',frame_test)
        video_write.write(frame_test)
        count+=1
        if cv2.waitKey(1) & 0xFF==27:
            print("Quitting the program")
            break
    except:
        count+=1
        timestamp = int(cam.get(cv2.CAP_PROP_POS_MSEC))  
        row_val=[timestamp,count,999999,999999]
        df.loc[len(df)] = row_val


df.to_csv('gaze_detection_20_03.csv',index=False)
cam.release()
video_write.release()
cv2.destroyAllWindows()