import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import uuid

def unsharp_masking(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return sharpened

path=r'E:\tensorrt\Trip\Experimenter_9110002_53.mp4'
cap=cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps=int(cap.get(cv2.CAP_PROP_FPS))
fileid=str(uuid.uuid4())

record_path=f'yolov8_objectDetect_SAHI_{fileid}.avi'
fourcc=cv2.VideoWriter_fourcc(*'mp4v')
resize_width=width//2
resize_height=height//2

video_write=cv2.VideoWriter(record_path, fourcc,fps, (resize_width,resize_height))

model_path = "weights\yolov8l.pt"
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=model_path,
    confidence_threshold=0.7,
    device="cuda",  # or 'cuda:0'
)


detection_list=['person','car','truck','traffic light','stop sign','motorcycle','bus','truck']
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        print("End of Video")
        break
    frame_scene=frame[:height//2,:width//2]
    frame_scene=unsharp_masking(frame_scene)

    result = get_sliced_prediction(
    frame_scene,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
        )
    
    object_prediction_list = result.object_prediction_list
    for object in object_prediction_list:
        score,bbox,category=object.score,object.bbox,object.category
        object_class=category.name
        conf=score.value
        bbox_list=bbox.to_xyxy()
        x0=int(bbox_list[0])
        y0=int(bbox_list[1])
        x1=int(bbox_list[2])
        y1=int(bbox_list[3])
        start_point=(x0,y0)
        end_point=(x1,y1)
        if object_class in detection_list:
            cv2.putText(frame_scene,object_class,(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
            cv2.rectangle(frame_scene,start_point,end_point,(255,0,0),2)

    cv2.imshow('test',frame_scene)
    video_write.write(frame_scene)
    if cv2.waitKey(1) & 0xFF==27:
        print("Quitting the program")
        break

cap.release()
video_write.release()
cv2.destroyAllWindows()