import cv2
import cupy as cp
from cupyx.scipy.ndimage import zoom as cp_zoom
import torch
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def mouse_callback(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(f'Mouse clicked at x={x}, y={y}')

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Check if the image is on GPU and transfer it to CPU if needed
    # if isinstance(img, cp.ndarray):
    #     img = cp.asnumpy(img)
    # Convert to CuPy array
    img_cp = cp.asarray(img)
    # current shape [height, width]
    shape = img_cp.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = cp.round(cp.array([shape[1], shape[0]]) * r).astype(int)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = cp.mod(dw, stride), cp.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0,0
        new_unpad = cp.array([new_shape[1], new_shape[0]])
        ratio = cp.array([new_shape[1] / shape[1], new_shape[0] / shape[0]])  # width, height ratios
    dw //= 2  # divide padding into 2 sides
    dh //= 2
    # Resize using CuPy
    # img_cp = cp.asnumpy(cp.interpolation.zoom(img_cp, (ratio[1], ratio[0], 1), order=1, mode='nearest'))
    img_cp = cp_zoom(img_cp, (ratio[1], ratio[0], 1), order=1, mode='nearest')
    top, bottom = int(cp.round(dh - 0.1)), int(cp.round(dh + 0.1))
    left, right = int(cp.round(dw - 0.1)), int(cp.round(dw + 0.1))
    # Add border using CuPy
    img_cp = cp.pad(img_cp, ((top, bottom), (left, right),(0,0)), mode='constant', constant_values=color[0])
    # Convert back to NumPy array
    # img_np = cp.asnumpy(img_cp)
    img_torch = torch.as_tensor(img_cp, device='cuda')
    return img_torch, img_torch.shape, (dw, dh)

def draw_facial_landmarks_on_image(rgb_image, detection_result):
  
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def eye_aspect_ratio_SPIGA(landmarks_int):
    """
    EAR Formula= EAR=||P2-P6||+||P3-P5||/2*(||P2-P4||)
    """
    
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    denominator_EAR_left= np.linalg.norm(landmarks_int[60] - landmarks_int[64])  # Left Eye
    denominator_EAR_right= np.linalg.norm(landmarks_int[68] - landmarks_int[72])  # Right Eye
    # ||P3-P5|| left
    left_p3_p5=np.linalg.norm(landmarks_int[62] - landmarks_int[66])
    right_p3_p5=np.linalg.norm(landmarks_int[70] - landmarks_int[74])
    #||P2-P6|| left
    left_61_67_dist=np.linalg.norm(landmarks_int[61] - landmarks_int[67])
    left_63_65_dist=np.linalg.norm(landmarks_int[63] - landmarks_int[65])

    right_69_75_dist=np.linalg.norm(landmarks_int[69] - landmarks_int[75])
    right_71_73_dist=np.linalg.norm(landmarks_int[71] - landmarks_int[73])

    left_p2_p6=(left_61_67_dist+left_63_65_dist)/2
    right_p2_p6=(right_69_75_dist+right_71_73_dist)/2

    EAR_left=((left_p2_p6+left_p3_p5)/(2*denominator_EAR_left))
    EAR_right=((right_p2_p6+right_p3_p5)/(2*denominator_EAR_right))

    return EAR_left,EAR_right