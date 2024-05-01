from ultralytics import YOLO
import cv2
import cvzone
import math
from openCV_projects.sort import *



cap = cv2.VideoCapture('C:/Users/bigya/PycharmProjects/env312/pythonProject/video/cars.mp4')

model = YOLO('yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
mask = cv2.imread('../openCV_projects/mask.png')

# tracking
tracker = Sort(max_age=20, min_hits = 3, iou_threshold=0.3)

limits = [423, 297, 673, 297]
totalCount = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)

    imgGraphics = cv2.imread('graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0,0))

    results = model(imgRegion, stream = True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(img, pt1 = (x1, y1), pt2 = (x2, y2), color = (0, 255,0), thickness = 2)

            #confidence
            conf = (math.ceil(box.conf[0]*100))/100
            # cvzone.putTextRect(img, f'{conf}', (max(0,x1), max(30,y1)))

            #class name
            cls = int(box.cls[0])
            if classNames[cls] == 'truck' or classNames[cls] == 'car' or classNames[cls] == 'motorbike' and conf > 0.3:
                # cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
                #cvzone.putTextRect(img, f'{classNames[cls]}  {conf}', pos = (max(0,x1), max(35, y1)), scale = 0.85, thickness = 1, offset= 1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))


    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color = (0,0,255), thickness = 5)

    for results in resultsTracker:
        x1, y1, x2, y2, id = results
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(results)
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(0,0, 255), thickness=2)
        cvzone.putTextRect(img, f'{int(id)}', pos=(max(0, x1), max(35, y1)),
                           scale=3, thickness=1, offset=5)

        cx, cy = x1 + (x2 - x1)/2, y1 + (y2 - y1)/2
        cx, cy = int(cx), int(cy)
        print(cx)
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-15< cy < limits[3] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), color=(0, 255, 0), thickness=5)

        # cvzone.putTextRect(img, f'count: {len(totalCount)}', (50, 50))
        cv2.putText(img, f'{len(totalCount)}', (200, 100), cv2.FONT_HERSHEY_TRIPLEX, 4 , (255, 50, 50))

    cv2.imshow('videoimage', img)
    # cv2.imshow('mask_image', imgRegion)
    cv2.waitKey(1)