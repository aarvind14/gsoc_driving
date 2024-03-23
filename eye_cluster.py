
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


df_eye=pd.read_csv(r'D:\pose_detect\trip_test\eye_features.csv')
df_slice=df_eye[['EAR_left','EAR_right']]

column_list=df_slice.columns
for column in column_list:
    sum_val=sum(np.isinf(df_slice[column]))
    if sum_val>0:
        df_slice[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        median = df_slice[column].median()
        df_slice[column].fillna(median, inplace=True)

X=df_slice.values
wcss = []
lable_list=[]

kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10,random_state=42)
kmeans.fit(X)
df_eye['eye_class']=kmeans.labels_
# lable_list.append(kmeans.labels_)
# wcss.append(kmeans.inertia_)

import cv2
path=r'Experimenter_9110002_53.mp4'
cap=cv2.VideoCapture(path)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=cap.get(cv2.CAP_PROP_FPS)
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
reduced_width=width//2
reduced_height=height//2
video_write_0=cv2.VideoWriter('class_eye_0.avi',fourcc,fps,(reduced_width,reduced_height))
video_write_1=cv2.VideoWriter('class_eye_1.avi',fourcc,fps,(reduced_width,reduced_height))
error_write=cv2.VideoWriter('error_frames.avi',fourcc,fps,(reduced_width,reduced_height))
frame_count=0
try:
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        frame_scene=frame[height//2:,:width//2]
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))


        if df_eye['eye_class'][frame_count]==0:
            video_write_0.write(frame_scene)

        elif df_eye['eye_class'][frame_count]==1:
            video_write_1.write(frame_scene)
        frame_count+=1
except Exception as e:
    print("Error")
    error_write.write(frame_scene)

    

cap.release()
error_write.release()
video_write_0.release()
video_write_1.release()
print("Done with the program")



  