# gsoc_driving
[Drive Link](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EixqFFFo0kBGlnpzkpn6Y-cBKI5rBXsKPe-XWgMPVJIvfg?e=Zodhb2)<br>
To look at the results you can click [*here*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EhGgjwxmA_JKowdUTe_77xkBhf6JXnCIZlRTqayXbLvM1Q?e=pdw0S1):


# Installation Instructions
To install SPIGA and the L2CS net look at the following repositories<br>
1) [**SPIGA**](https://github.com/andresprados/SPIGA) <br>
In case of pickle error for SPIGA download the spiga model from the [drive](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EixqFFFo0kBGlnpzkpn6Y-cBKI5rBXsKPe-XWgMPVJIvfg?e=Zodhb2) and put it in the directory *spiga/models/weights*

2) [**L2CS**](https://github.com/Ahmednull/L2CS-Net) Go to the repository and follow the installation instructions
   
3) Run pip install -r requirements.txt

# How to Get results:
1) Run *crash_detect.py* to get the when the car is at risk of collision with objects. The ROI which is selected can be made more or less aggressive by choice.<br>
To look directly at the results click [*here*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/Efm1PffV65tPj66fBJC6LngBnRf8-cy0_CoJxPxzoF9gow?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=i7xSVc)

2) Run head position_df to get a dataframe of the head orientation at ech frame and a video. Then with the dataframe you get with this you can run head-pose_cluster.py to get the cluster of videos for head pose. Results can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EleFjDt6_zpAtKrGeyXYS88BV099M2hrqo1vhVy_qcK-HQ?e=9vyweE)

3) Run ear_drowsiness_df.py to get the EAR dataframe for left and right eye and then run eye_cluster.py to get the drowsiness and blink frames. Results can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EkdF0fOqWOVOnWhygIfMYMcBq4M2vzuLH9JeuPBE5ZQF_g?e=on4Csk)

4) Run poseNASonehand.py to detect when the driver is driving using only one hand. Result can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EYzNUIjl4OlPgSeiqXlx7WkBB4AfcjAUBuBzEAvmnyLU6A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=WRch0e)

5) For running the emotion detection results run emotion_recog_df.py. The reults can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/ETlTr0MzkcxJmTAX4F01nOQBYsK1j2b_33eX9pj23fzUdw?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=qHS2Qs)

6) For object detection and tracking using yolov8 run track_yolov8.py. Can change between the pretained model and the custom alabama model
   
7) For running object detection and tracking with YOLOv8+SAHI+OCSORT tracking algorithm run trackOC.py.[*result for pretained model*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EbE8tTguNzRMrq4tXkmjX1kBiEcV_HY3qPArUyhwYkTeBA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=i6C7F0)

8) Run data_cleaning.py to make the number of frames and the readings equal. It also drops the empty columns and columns with only one unique value. Some reults related to it can be found in *crash.ipynb*

   



   

