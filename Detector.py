from ultralytics import YOLO
import cv2
import math
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image,ImageTk
import threading
class Detector:
    def __init__(self,ep_robot,nn,panel,device='cpu'):
        self.ep_robot = ep_robot
        #self.display_thread = threading.Thread(target=self.start_stream)
        self.panel = panel
        self.device = device
        self.stop_stream = False
        self.ep_camera = self.ep_robot.camera
        self.nn=nn
        self.load_model()
    def load_model(self):
        #self.nn=nn
        if self.nn=='yolo':
            self.model=YOLO("yolo-Weights/yolov8n.pt").to(self.device)
            self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
        elif self.nn=='mobilenet':
            prototxt = "MobileNetSSD_deploy.prototxt"
            caffe_model = "MobileNetSSD_deploy.caffemodel"
            self.model = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)
            self.classNames = {0: 'background',
                  1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
                  5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
                  10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
                  14: 'motorbike', 15: 'person', 16: 'pottedplant',
                  17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
            if self.device == 'cuda':
                self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif self.nn=="custom":
            self.model=YOLO("yolo-Weights/custom.pt").to(self.device)
            self.classNames=self.model.names

    def process_image(self,img):
        if self.nn=='yolo':
            results = self.model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    cls = int(box.cls[0])
                    print("Class name -->", self.model.names[cls])

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.putText(img, self.model.names[cls], org, font, fontScale, color, thickness)
                    cv2.putText(img, str(confidence), [x2,y1+10], font, fontScale, color, thickness)
            return img
        elif self.nn=='mobilenet':
            frame = img.reshape(360, 640, 3)
            width = frame.shape[1]
            height = frame.shape[0]
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1 / 127.5, size=(300, 300), mean=(127.5, 127.5, 127.5),
                                         swapRB=True, crop=False)
            self.model.setInput(blob)
            detections = self.model.forward()
            print(detections.shape[2])
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    class_id = int(detections[0, 0, i, 1])
                    x_top_left = int(detections[0, 0, i, 3] * width)
                    y_top_left = int(detections[0, 0, i, 4] * height)
                    x_bottom_right = int(detections[0, 0, i, 5] * width)
                    y_bottom_right = int(detections[0, 0, i, 6] * height)

                    cv2.rectangle(frame, (x_top_left, y_top_left), (x_bottom_right, y_bottom_right),
                                  (0, 255, 0))

                    if class_id in classNames:
                        label = self.classNames[class_id] + ": " + str(confidence)
                        (w, h), t = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y_top_left = max(y_top_left, h)
                        cv2.rectangle(frame, (x_top_left, y_top_left - h),
                                      (x_top_left + w, y_top_left + t), (0, 0, 0), cv2.FILLED)
                        cv2.putText(frame, label, (x_top_left, y_top_left-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
            return frame
        elif self.nn=="custom":
            results = self.model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    cls = int(box.cls[0])
                    print("Class name -->", 'robot')

                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 0.5
                    color = (0, 255, 0)
                    thickness = 2
                    cv2.putText(img, 'robot', org, font, fontScale, color, thickness)
                    cv2.putText(img, str(confidence), [x2, y1 + 10], font, fontScale, color, thickness)
            return img
    def start_stream(self):
        self.ep_camera.start_video_stream(display=False, resolution="360p")
        while not self.stop_stream:
            img = self.ep_camera.read_cv2_image(strategy='newest')
            img = self.process_image(img)
            self.display_image(img)
        self.ep_camera.stop_video_stream()
    def display_image(self,img):
        b, g, r = cv2.split(img)
        imgpil = cv2.merge((r, g, b))
        imgpil = Image.fromarray(imgpil)
        imgtk = ImageTk.PhotoImage(image=imgpil)
        self.panel.imgtk = imgtk
        self.panel.config(image=imgtk)
