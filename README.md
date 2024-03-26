# gsoc_driving
[Drive Link](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EixqFFFo0kBGlnpzkpn6Y-cBKI5rBXsKPe-XWgMPVJIvfg?e=Zodhb2)<br>
To look at the results you can click [*here*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EhGgjwxmA_JKowdUTe_77xkBhf6JXnCIZlRTqayXbLvM1Q?e=pdw0S1):

# Results Discussion:
## Drowsiness detection
1) Drowsiness of the driver using the EAR(Eye aspect ratio) for every frame of the video and divided the video into 2 clusters using K-means clustering. Then drowsiness was quantified if the eyes were closed for more than 5 frames. The results were saved in the [file](https://indianinstituteofscience-my.sharepoint.com/:x:/g/personal/adityaarvind_iisc_ac_in/EX4ZxkvdFpdPp-ocuwcU82kBA866rFa5Onkj81sdSP6BJw?e=xyqisa) *dataframe_cluster/drowsy_frames.csv*. The video snippets can be seen in the ear_cluster folder.
2) Drowsiness gaze attention: Applied a gaze detection script on the video and saved the results in [this](https://indianinstituteofscience-my.sharepoint.com/:x:/g/personal/adityaarvind_iisc_ac_in/EcnNefz9RI1Ah8eIEvMuBJIBd7mbvSzjaWYnjVZmHwArfQ?e=QSrg2D) csv file. Then after seperating the drowsy frames by headpose values ,it was noticed that when the driver was drowsy she was not focused on the road and was looking at the steering of the vehicle for a portion of the time.
## Risky driving behaviour
1) Risky driving behaviour can be checked by how long is the driver not paying attention straight ahead and is looking somewhere else. Head pose estimation was used to get the pitch, yaw and roll values of the driver and then K-Means clustering was applied and the frames were divided into 4 clusters.Largest cluster is for the correct driving behaviour. The videos for the clusters can be found in head_pose_cluster.
2) After clustering the frames 4 csv files were created which contain the startpoint,endpoint and the duration for which the driver was in that pose. Just like with drowsiness if the eyes of the driver are away from the road for a prolonged period of time. It can be classified as dangerous behaviour. Results can be found dataframe_cluster folder.
3) Can also be used as a measure the attentiveness of the driver

## Annotation of Objects of interest
1) Firstly the pretarined YOLOv8 model with SAHI was used to test the video in objectDetectionSAHI.py. Then OCSort tracking was used alongwith YOLOv8_SAHI to effectively track the objects of interest. The results can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EbE8tTguNzRMrq4tXkmjX1kBiEcV_HY3qPArUyhwYkTeBA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=OsF2cO)

2) A custom object detection model was trained on the [alabama ATI dataset](https://universe.roboflow.com/alabama-transport-insititue-tbwq8/ati-yolov8). An arificial class for trash can was added to detect it in the video.

3) Using these two models we can identify all the objects of interest in the frame like in Collision_detection.avi
*Note*: This video does not use SAHI and just uses plain object detection.

## Risky Scenario and Agressive driver
1) Code can be found in crash_detect.py. An arbitrary ROI was chosen for the detection of sudden objects like trash can that come into the frame. When they come into the boundary of ROI their bounding box becomes red meaning our program has detected them. Currently I do not filter for vehicles like car or truck thus false alerts come for them. But similar concept can be used to detect tailgating by the driver. The results can be seen in this [video](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/Efm1PffV65tPj66fBJC6LngBnRf8-cy0_CoJxPxzoF9gow?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=nhLxHR)

## Miscelllanious Results
### Gaze detection & One hand driving detection
The results can be seen in the videos [gaze_detect.avi](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EQjio4EcFWtNpeXZKAlovLYBy9SocVHh9lupRgYYGMwkDg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=vjF7NG) and [driver distraction pose](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EYzNUIjl4OlPgSeiqXlx7WkBB4AfcjAUBuBzEAvmnyLU6A?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=jTjYlC)

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
   
7) For running object detection and tracking with YOLOv8+SAHI+OCSORT tracking algorithm run trackOC.py. [(*result for pretained model*)](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EbE8tTguNzRMrq4tXkmjX1kBiEcV_HY3qPArUyhwYkTeBA?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=i6C7F0)

8) Run data_cleaning.py to make the number of frames and the readings equal. It also drops the empty columns and columns with only one unique value. Some reults related to it can be found in *crash.ipynb*

9) Run gaze.py inside the GAZE_detection folder to get a dataframe of the gaze of the driver at every timestep. Result can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:v:/g/personal/adityaarvind_iisc_ac_in/EQjio4EcFWtNpeXZKAlovLYBy9SocVHh9lupRgYYGMwkDg?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=mkWJmz)

# Results from dataframe analysis
1) Dataframes obtained from above scripts were used to categorize drowsy and distracted behaviour of the driver. The code for that can be found in drowsy.ipynb [*results*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/Eqfa-8ZzrOdNrH1JFYZCGn4BiJIzKGC7H1H9-AaR1rElZg?e=j8fQ23)
2) Code for Gaze attention during drowsiness can be found in data_analysis_merged.ipynb.Some graphs related to it can be found [*here*](https://indianinstituteofscience-my.sharepoint.com/:f:/g/personal/adityaarvind_iisc_ac_in/EjDmWUW4w8BAmqsoJdcOip8BtosZUn8WuuYz2BAZn_WPhQ?e=geJ0OO)


   



   

