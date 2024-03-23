import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from ultralytics import YOLO
import uuid
from trip_utils import vidStream
import numpy as np
from functions_utils import letterbox
import torch

resize_width=640
resize_height=480
device='cuda' if torch.cuda.is_available() else 'cpu'

weight_path=r'E:\tensorrt\Trip\weights\yolov8x.pt'
path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'
fileid=str(uuid.uuid4())
record_path=f'yolov8_object_{fileid}.avi'
fourcc=cv2.VideoWriter_fourcc(*'XVID')
# Create model
model=YOLO(weight_path)
camera=cv2.VideoCapture(path)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
fps=int(camera.get(cv2.CAP_PROP_FPS))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

frame_y= height//2    # Frame shape values and precalculation of the mask
frame_x= width//2
n_channels=3
frame_shape=np.array([frame_x,frame_y])   #To get the frame shape
video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

while camera.isOpened():
    # try:
        ret,frame=camera.read()
        if not ret:
            print("End of Video")
            break
        frame_scene=frame[:height//2,:width//2]
        frame_show=frame_scene.copy()
        frame_scene=torch.from_numpy(frame_scene)
        frame_scene=frame_scene.to(device)

        frame_scene,shape,ignore2=letterbox(frame_scene)
        letterbox_shape=np.array([shape[1],shape[0]])  # matrix representation to opencv representation
        frame_scene=frame_scene/255.0
        frame_scene = frame_scene[:, :, [2, 1, 0]].transpose(0, 1).transpose(0, 2).contiguous()
        frame_scene = frame_scene.unsqueeze(0)
        frame_scene=frame_scene.to(dtype=torch.float16)
        results=model.predict(frame_scene)
        result=results[0]
        for box in result.boxes:
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            start_point = np.array((cords[0],cords[1]))/letterbox_shape   # First x then y (Normalising the points)
            start_point=start_point*frame_shape
            start_point=start_point.astype(int,copy=False)
            end_point = np.array((cords[2], cords[3]))/letterbox_shape
            end_point=end_point*frame_shape     # Getting the points into the final frame shape
            end_point=end_point.astype(int,copy=False)
            if conf > 0.7:  # This value needs to be played with
                frame_show = cv2.rectangle(frame_show, start_point , end_point, (255,0,0), 2) # blue color

        frame_show=cv2.resize(frame_show,(resize_width,resize_height))
        cv2.imshow('frame_scene',frame_show)
        video_write.write(frame_show)
        if cv2.waitKey(1) & 0xFF==27:
            print("Quitting the program")
            break
    # except Exception as e:
    #     print(e)
camera.release()
video_write.release()
cv2.destroyAllWindows()