from multiprocessing import dummy
import cv2 as cv
import torch
import time
import numpy as np


def score_frame(frame,model):
    #print(frame.shape)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    cv.imshow("Score input",frame)
    #print(frame)
    results=model(frame)
    labels=results.xyxyn[0][:,-1].numpy()
    coord=results.xyxyn[0][:,:-1].numpy()
    return results,labels,coord

def plot_bounding_boxes(model,result,labels,coord,frame):
    num_items=len(labels)
    x_shape,y_shape=frame.shape[1],frame.shape[0]
    for i in range(num_items):
        row=coord[i]
        if row[4]<0.2: #confidence threshold
            continue
        x1=int(row[0]*x_shape)
        y1=int(row[1]*y_shape)
        x2=int(row[2]*x_shape)
        y2=int(row[3]*y_shape)
        bounding_box_colour_in_bgr=(0,255,0)
        classnames=model.names
        label_font=cv.FONT_HERSHEY_COMPLEX
        cv.rectangle(frame,(x1,y1),(x2,y2),bounding_box_colour_in_bgr,1) #bounding box drawer
        str_for_bounding_box="Class: "+str(classnames[int(labels[i])])+"   "+"Confidence: "+str(np.round(row[4],5))
        cv.putText(frame,str_for_bounding_box,(x1,y1),label_font,0.5,(255,255,255),1) #class names
    return frame

yolo=torch.hub.load('ultralytics/yolov5', 'yolov5n',pretrained=True)

webcam=cv.VideoCapture(0)

if not webcam.isOpened():
    print("Could not open webcam")
    exit()

while(True):
    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    start_time=time.time()
    ret,frame=webcam.read()
    if not ret:
        print("Webcam not working")
        break
    infer_frame=cv.resize(frame,(640,640))
    #cv.imshow('input',frame)
    #cv.imshow('infer_frame',infer_frame)
    result,label,coord=score_frame(infer_frame,yolo)
    out_frame=plot_bounding_boxes(yolo,result,label,coord,infer_frame)
    end_time=time.time()
    fps=1/np.round((end_time-start_time),5)
    str_for_fps='FPS: '+str(fps)
    cv.putText(out_frame,str_for_fps,(10,10),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv.imshow('Output',out_frame)

print('Done')
