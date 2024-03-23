import cv2
from threading import Thread
import threading
import numpy as np

class vidStream:
    def __init__(self,vid_path,record=True,record_path='out_vid.avi',fourcc=cv2.VideoWriter_fourcc(*'mp4v'),width=0,height=0):
        self.path=vid_path
        self.cam=cv2.VideoCapture(vid_path)
        self.stop_flag = threading.Event()
        self.thread=Thread(target=self.readFrame,args=())
        self.thread.daemon=True
        # self.thread.start()
        self.ret=True
        self.record=record
        self.fps=int(self.cam.get(cv2.CAP_PROP_FPS))
        
        self.height=int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width=int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame=np.zeros((self.height,self.width, 3), dtype=np.uint8)
        self.frame_prev=np.zeros((self.height,self.width, 3), dtype=np.uint8)
        self.frame_list=[(self.ret,self.frame)]
        if self.record:
            self.record_path=record_path
            self.fourcc=fourcc
            if width:
                self.VideoWriter=cv2.VideoWriter(self.record_path, self.fourcc, self.fps, (width,height))
            else:
                self.VideoWriter=cv2.VideoWriter(self.record_path, self.fourcc, self.fps, (self.width, self.height))

    def startStream(self):
        self.thread.start()

    def readFrame(self):
        while not self.stop_flag.is_set():
            self.ret,self.frame=self.cam.read()
            if not (np.array_equal(self.frame,self.frame_prev)):
                self.frame_list.append((self.ret,self.frame))
                self.frame_prev=self.frame
            else:
                continue

    def getFrame(self):
        return self.frame_list.pop(0)
    
    def endStream(self):
        self.stop_flag.set()
        self.thread.join()
        self.VideoWriter.release()
        self.cam.release()

class cameraVideo:
    def __init__(self,path):
        self.cam=cv2.VideoCapture(path)
        self.thread=Thread(target=self.fetch,args=())
        self.thread.daemon=True
        self.stop_flag = threading.Event()
        self.thread.start()
        self.ret=True
    def fetch(self):
        while not self.stop_flag.is_set():
            self.ret,self.frame=self.cam.read()
            self.timestamp=int(self.cam.get(cv2.CAP_PROP_POS_MSEC))
    def getFrame(self):
        return self.ret,self.frame