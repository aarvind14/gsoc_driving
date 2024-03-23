import cv2
import numpy as np
import super_gradients
import uuid
import matplotlib.path as mpltPath

model_pose = super_gradients.training.models.get("yolo_nas_pose_l", pretrained_weights="coco_pose").cuda()
path=r'E:\tensorrt\Trip\Experimenter_9110002_53.mp4'
steer=np.array([[0.38645833, 0.87777778],[0.409375  , 0.73703704],[0.44375 , 0.6],\
       [0.51666667, 0.53703704],[0.59479167, 0.46481481],[0.709375  , 0.41296296],\
       [0.78854167, 0.43148148],[0.83541667, 0.52777778],[0.85416667, 0.59259259],\
       [0.73958333, 0.74259259],[0.703125  , 0.83148148],[0.665625  , 0.87222222],\
       [0.4875    , 0.87777778],[0.43229167, 0.87037037]])
cap=cv2.VideoCapture(path)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
file_id=str(uuid.uuid4())
record_path=f'Driver_distraction_pose_{file_id}.avi'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
resized_width=width//2
resized_height=height//2

new_ROI=steer*np.array([resized_width,resized_height])
new_ROI=new_ROI.astype('int')
path = mpltPath.Path(new_ROI)
video_write=cv2.VideoWriter(record_path,fourcc, fps, (resized_width, resized_height))

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("End of Video")
        break
    frame_pose=frame[height//2:,:width//2]   # Bottom left front camera
    # frame_pose=frame[:height//2,width//2:]    # Top right dark camera
    # Converting to bgr improves the accuracy
    test=cv2.cvtColor(frame_pose,cv2.COLOR_BGR2RGB)
    results=model_pose.predict(test)
    poses=results.prediction.poses
    bbox=results.prediction.bboxes_xyxy
    conf=results.prediction.scores
    if len(conf)<1:
        video_write.write(frame_pose)
        cv2.imshow('test_conf',frame_pose)
        continue
    max_val=np.argmax(conf)

    x0=int(bbox[max_val][0])
    y0=int(bbox[max_val][1])
    x1=int(bbox[max_val][2])
    y1=int(bbox[max_val][3])

    cv2.rectangle(frame_pose,(x0,y0),(x1,y1),(220,255,0),2)

    for idx,keypoint in enumerate(poses[max_val]):
        x,y,keypoint_conf=keypoint
        x=int(x)
        y=int(y)
        if keypoint_conf>0.3:
            cv2.putText(frame_pose,str(idx),((x-10),(y-10)),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2) # Putting the keypoints with number on the image
            cv2.circle(frame_pose,(x,y),2,(0,255,0),-1)
            if idx in [9,10]:  # pose points for the left and right wrist
                inside_points = path.contains_points([np.array([x,y])])
                if not inside_points:
                    cv2.putText(frame_pose,"One hand driving",(10,90),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)

    
    # cv2.imshow('test',frame_bottom)
    video_write.write(frame_pose)
    cv2.imshow('test2',frame_pose)
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break
cap.release()
video_write.release()
cv2.destroyAllWindows()