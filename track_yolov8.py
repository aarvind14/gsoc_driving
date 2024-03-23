import cv2
from ultralytics import YOLO
import uuid

# Default tracker is BOT-SORT

model_path=r'weights\yolov8x.pt'
path=r'Experimenter_9110002_53.mp4'
cap=cv2.VideoCapture(path)
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2
fileid=str(uuid.uuid4())
record_path=f'yolov8_track_{fileid}.avi'
video_write=cv2.VideoWriter(record_path, fourcc,fps,(resize_width,resize_height))
model=YOLO(model_path)

while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("End of Video")
        break
    frame_scene=frame[:height//2,:width//2]
    # track Objects
    results=model.track(frame_scene,persist=True,conf=0.7)
    class_dict=results[0].names
    bboxes=results[0].boxes
    if bboxes is not None:
        for box in bboxes:
            #Getting the start and end points for the boxes (x0,y0) starting point
            x0,y0,x1,y1=box.xyxy.cpu().numpy().astype('int')[0]
            start_point=(x0,y0)
            end_point=(x1,y1)
            cv2.rectangle(frame_scene,start_point,end_point,(255,0,0),2)
            box_class_num=box.cls.cpu().numpy().astype('int')[0]
            box_class=class_dict[box_class_num]
            cv2.putText(frame_scene,box_class,((x0+20),(y0-10)),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,255),2)
            
            if box.id is not None:
                id_num=box.id.cpu().numpy().astype('int')[0]
                cv2.putText(frame_scene,str(id_num),((x0-10),(y0-10)),cv2.FONT_HERSHEY_SIMPLEX,0.75,(0,0,128),2)
           
    video_write.write(frame_scene)
    cv2.imshow('track',frame_scene)
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break
cap.release()
video_write.release()
cv2.destroyAllWindows()