import cv2
import numpy as np
from ultralytics import YOLO
import uuid
from shapely.geometry import LineString

def line_intersects_quadrilateral(line_coords, quad_coords):
    line_left = LineString(line_coords[0])
    line_right = LineString(line_coords[1])
    line_top=LineString(line_coords[2])
    shift_cords = np.roll(quad_coords, -2)

    # Check if the line intersects any edge of the quadrilateral
    for edge in zip(quad_coords,shift_cords):
        edge_line = LineString(edge)
        if line_left.intersects(edge_line):
            return True
        if line_right.intersects(edge_line):
            return True
        if line_top.intersects(edge_line):
            return True
    return False

model_path=r'E:\tensorrt\Trip\weights\yolov8x.pt'
model_custom=r'E:\tensorrt\Trip\models\alabama_custom.pt'
model=YOLO(model_path)
model_custom=YOLO(model_custom)
coco_dict=model.names
custom_dict=model_custom.names
path=r'E:\tensorrt\TRIP\Experimenter_9110002_53.mp4'
cam=cv2.VideoCapture(path)
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cam.get(cv2.CAP_PROP_FPS))


new_ROI=np.array([[0.19375,0.99259259],
       [0.47395833,0.56111111],
       [0.54375,0.56111111],
       [0.96770833, 0.99259259]])
frame_size=np.array([width//2,height//2])
new_ROI=new_ROI*frame_size
new_ROI=new_ROI.astype('int')

file_id=str(uuid.uuid4())
record_path=f'Collision_detection_{file_id}.avi'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2 
video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

while cam.isOpened():
    ret,frame=cam.read()
    if not ret:
        print("End of Video")
        break
    # print(frame.shape)
    frame_scene=frame[:height//2,:width//2]
    frame_bottom=frame[height//2:,:width//2]
    
    results=model.predict(frame_scene,conf=0.7)
    results_custom=model_custom.predict(frame_scene,conf=0.3)
    result_custom=results_custom[0]
    result=results[0]
    for detect in result:
        boxes=detect.boxes.xyxy.cpu().numpy().astype('int')[0]
        class_name=detect.boxes.cls.cpu().numpy().astype('int')[0]
        class_name=coco_dict[class_name]
        x0=boxes[0]
        y0=boxes[1]
        start_point=(x0,y0)
        x1=boxes[2]
        y1=boxes[3]
        end_point=(x1,y1)
        leftLine=[(x0,y0),(x0,y1)]
        rightLine=[(x1,y0),(x1,y1)]
        topLine=[(x0,y0),(x1,y0)]
        line_cords_custom=[leftLine,rightLine,topLine]
        intersect=line_intersects_quadrilateral(line_cords_custom,new_ROI)

        if intersect:
            cv2.rectangle(frame_scene,start_point,end_point,(0,0,255),2)
            cv2.putText(frame_scene,'Danger',(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,120,0),2)
        else:
            cv2.rectangle(frame_scene,start_point,end_point,(0,255,0),2)
            cv2.putText(frame_scene,str(class_name),(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,120,0),2)

    for detect in result_custom:
        boxes_custom=detect.boxes.xyxy.cpu().numpy().astype('int')[0]
        class_name_custom=detect.boxes.cls.cpu().numpy().astype('int')[0]
        class_name_custom=custom_dict[class_name_custom]

        x0=boxes_custom[0]
        y0=boxes_custom[1]
        start_point=(x0,y0)
        x1=boxes_custom[2]
        y1=boxes_custom[3]
        end_point=(x1,y1)

        leftLine=[(x0,y0),(x0,y1)]
        rightLine=[(x1,y0),(x1,y1)]
        topLine=[(x0,y0),(x1,y0)]
        line_cords_custom=[leftLine,rightLine,topLine]

       
        intersect=line_intersects_quadrilateral(line_cords_custom,new_ROI)
        if intersect:
            cv2.rectangle(frame_scene,start_point,end_point,(0,0,255),2)
            cv2.putText(frame_scene,'Danger',(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,120,0),2)
        else:
            cv2.rectangle(frame_scene,start_point,end_point,(0,255,0),2)
            cv2.putText(frame_scene,str(class_name_custom),(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,120,0),2)


    
    
    frame_scene = cv2.polylines(frame_scene, pts=[new_ROI], color=(60,255,255), isClosed=True,thickness=2)
    video_write.write(frame_scene)
    cv2.imshow('frame_scene',frame_scene)
    # cv2.setMouseCallback('frame_scene',mouse_callback)
 
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break
cam.release()
video_write.release()
cv2.destroyAllWindows()