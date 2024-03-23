
import pandas as pd
from sklearn.cluster import KMeans

df_head=pd.read_csv('head_roll.csv')
df_eye=pd.read_csv('eye_features.csv')
merged_df = pd.merge(df_eye, df_head, on='Timestamp')
merged_df['IRIS_left_x']=merged_df['IRIS_left_x']/960   # Dividing by the width of the video
merged_df['IRIS_left_y']=merged_df['IRIS_left_y']/540   # Dividing by the height of the video
merged_df['IRIS_right_x']=merged_df['IRIS_right_x']/960 # Dividing by the width of the video
merged_df['IRIS_right_y']=merged_df['IRIS_right_y']/540 # Dividing by the height of the video

# Min-Max Scaling
merged_df['Pitch_Norm'] = (merged_df['Pitch'] - merged_df['Pitch'].min()) / (merged_df['Pitch'].max() - merged_df['Pitch'].min())
merged_df['Roll_Norm'] = (merged_df['Roll'] - merged_df['Roll'].min()) / (merged_df['Roll'].max() - merged_df['Roll'].min())
merged_df['Yaw_Norm'] = (merged_df['Yaw'] - merged_df['Yaw'].min()) / (merged_df['Yaw'].max() - merged_df['Yaw'].min())

df_slice=merged_df[['IRIS_left_x','IRIS_left_y','IRIS_right_x','IRIS_right_y','Pitch_Norm','Yaw_Norm','Roll_Norm']]

X=df_slice.values
wcss = []
lable_list=[]

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10,random_state=42)
kmeans.fit(X)
merged_df['eye_class']=kmeans.labels_
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
video_write_0=cv2.VideoWriter('class_0.avi',fourcc,fps,(reduced_width,reduced_height))
video_write_1=cv2.VideoWriter('class_1.avi',fourcc,fps,(reduced_width,reduced_height))
video_write_2=cv2.VideoWriter('class_2.avi',fourcc,fps,(reduced_width,reduced_height))
video_write_3=cv2.VideoWriter('class_3.avi',fourcc,fps,(reduced_width,reduced_height))
error_write=cv2.VideoWriter('error_frames.avi',fourcc,fps,(reduced_width,reduced_height))
frame_count=0
try:
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        frame_scene=frame[height//2:,:width//2]
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))


        if merged_df['eye_class'][frame_count]==0:
            video_write_0.write(frame_scene)

        elif merged_df['eye_class'][frame_count]==1:
            video_write_1.write(frame_scene)

        elif merged_df['eye_class'][frame_count]==2:
            video_write_2.write(frame_scene)

        elif merged_df['eye_class'][frame_count]==3:
            video_write_3.write(frame_scene)
        
        frame_count+=1
except Exception as e:
    print("Error")
    error_write.write(frame_scene)

    

cap.release()
error_write.release()
video_write_0.release()
video_write_1.release()
video_write_2.release()
video_write_3.release()
print("Done with the program")



  