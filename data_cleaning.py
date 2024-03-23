import pandas as pd
import numpy as np
# Specify the path to your .dat file
file_path = r'E:\tensorrt\TRIP\Experimenter_9110002_53.dat'
data_trip = pd.read_csv(file_path, sep=' ')

unique_elements=[]
less_10_col_list=[] # The list of columns with less than 10 unique elements
for column in data_trip.columns:
    unique_count=data_trip[column].nunique()
    unique_elements.append(unique_count)
    if unique_count<=1:
        less_10_col_list.append(column)

data_trip.drop(columns=less_10_col_list,inplace=True)
max_val=0
count=0
mediaTime_list=data_trip['MediaTime'].unique()
df_difference=pd.DataFrame(columns=data_trip.columns)

for time in mediaTime_list:
    df_slice=data_trip[data_trip['MediaTime']==time].reset_index(drop=True)
    for i in range(0,len(df_slice)-1):
        diff=abs(df_slice.iloc[i+1]-df_slice.iloc[i])
        df_difference = pd.concat([df_difference, diff.to_frame().T], ignore_index=True)
    max_val=max(max_val,len(df_slice))
mediaTime_list=data_trip['MediaTime'].unique()

df_mediaTime=pd.DataFrame(columns=data_trip.columns)
for time in mediaTime_list:
    df_slice=data_trip[data_trip['MediaTime']==time].reset_index(drop=True)
    condense=df_slice.mean()
    df_mediaTime = pd.concat([df_mediaTime, condense.to_frame().T], ignore_index=True)

# Checking for missing seconds
df_mediaTime['MediaTime'].max()
frame_list=[]
for i in range(1,int(df_mediaTime['MediaTime'].max())+1):
    total=len(df_mediaTime[(df_mediaTime['MediaTime']>i-1) & (df_mediaTime['MediaTime']<i)])
    frame_list.append((i,total))
sorted_list=sorted(frame_list,key=lambda x: x[1])
filtered_list = [tup for tup in sorted_list if tup[1] <30]  
miss_seconds_list=[tup[0] for tup in filtered_list]

index_list=df_mediaTime[df_mediaTime['FollowCarBrakingStatus']>0].index
for i in index_list:
    df_mediaTime.loc[i,'FollowCarBrakingStatus']=1

df_mediaTime['Gear']=df_mediaTime['Gear'].astype('int')
df_mediaTime.to_csv('cleaned_experiments.csv',index=False)

