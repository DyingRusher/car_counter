from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/car_counter/video.mp4')
# cap.set(3,640)
# cap.set(4,480)
mask = cv2.imread('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/car_counter/mask.png')
model = YOLO('C:/Users/arjav/OneDrive/Desktop/object_detection (1)/yolov8n.pt')
yolo_map = {0: 'person',
 1: 'bicycle',
 2: 'car',
 3: 'motorcycle',
 4: 'airplane',
 5: 'bus',
 6: 'train',
 7: 'truck',
 8: 'boat',
 9: 'traffic light',
 10: 'fire hydrant',
 11: 'stop sign',
 12: 'parking meter',
 13: 'bench',
 14: 'bird',
 15: 'cat',
 16: 'dog',
 17: 'horse',
 18: 'sheep',
 19: 'cow',
 20: 'elephant',
 21: 'bear',
 22: 'zebra',
 23: 'giraffe',
 24: 'backpack',
 25: 'umbrella',
 26: 'handbag',
 27: 'tie',
 28: 'suitcase',
 29: 'frisbee',
 30: 'skis',
 31: 'snowboard',
 32: 'sports ball',
 33: 'kite',
 34: 'baseball bat',
 35: 'baseball glove',
 36: 'skateboard',
 37: 'surfboard',
 38: 'tennis racket',
 39: 'bottle',
 40: 'wine glass',
 41: 'cup',
 42: 'fork',
 43: 'knife',
 44: 'spoon',
 45: 'bowl',
 46: 'banana',
 47: 'apple',
 48: 'sandwich',
 49: 'orange',
 50: 'broccoli',
 51: 'carrot',
 52: 'hot dog',
 53: 'pizza',
 54: 'donut',
 55: 'cake',
 56: 'chair',
 57: 'couch',
 58: 'potted plant',
 59: 'bed',
 60: 'dining table',
 61: 'toilet',
 62: 'tv',
 63: 'laptop',
 64: 'mouse',
 65: 'remote',
 66: 'keyboard',
 67: 'cell phone',
 68: 'microwave',
 69: 'oven',
 70: 'toaster',
 71: 'sink',
 72: 'refrigerator',
 73: 'book',
 74: 'clock',
 75: 'vase',
 76: 'scissors',
 77: 'teddy bear',
 78: 'hair drier',
 79: 'toothbrush'}

while True:
    
    suc ,img = cap.read()
    imgreg = cv2.bitwise_and(img,mask)
    res = model(imgreg,stream = True)
    # print(res)
    
    for r in res:
        boxes = r.boxes
        for b in boxes:
            #for cv2
            x1,y1,x2,y2 = b.xyxy[0]
            x1,y1,x2,y2  = int(x1),int(y1),int(x2),int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(34,23,133),3)
            # print(x1,y1,x2,y2)
            
            bbox = int(x1),int(y1),int(x2-x1),int(y2-y1)
            cvzone.cornerRect(img,bbox,l=3)
            conf = (math.ceil(b.conf*100))/100 #0.343553 to 0.34
            cla = int(b.cls[0])
            # print(yolo_map[cla])
            if(yolo_map[cla] == 'truck' or yolo_map[cla] == 'bus' or yolo_map[cla] == 'motorcycle' or yolo_map[cla] == 'car'):
                cvzone.putTextRect(img,f'{conf} {yolo_map[cla]}',(max(0,x1),max(30,y1)),scale=1,thickness=1,offset=1)
                cvzone.cornerRect(img,bbox,l=3)
            # cv2.putText(img,str(conf),(int(x1),int(y2-10)),cv2.FONT_HERSHEY_SIMPLEX,1,(25,25,134),2,cv2.LINE_AA)
            
    cv2.imshow("sds",img)
    intr = cv2.waitKey(0)
    if intr & 0xFF == ord('q') :
        break