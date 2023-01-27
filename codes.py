import cv2
import numpy as np

cap=cv2.VideoCapture('video6.mp4')

classesFiles='D:/robo/new/coco.names.txt'
classNames=[]
with open(classesFiles,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')



whT=320
confThreshold=0.5
nmsThreshold=0.3


modelConfiguration='yolov3.cfg'
modelWeights='yolov3.weights'

net=cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(outputs,img):
    hT,wT,cT=img.shape
    bbox=[]
    classIds=[]
    confs=[]

    for output in outputs:
        for det in output:
            scores=det[5:]
            classId=np.argmax(scores)
            confidence=scores[classId]

            if confidence > confThreshold:
                w,h= int(det[2]*wT), int(det[3]*hT)
                x,y= int(det[0]*wT - w/2),int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices=cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    print(indices)
    if len(indices) > 0:
      for i in indices.flatten():
        
        x, y, w, h = bbox[i]
        #x,y,w,h=box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX , 0.6,(255,0,255),2)



while True:
    success,img= cap.read()

    blob=cv2.dnn.blobFromImage(img,1/255,(320,320),[0,0,0],1,crop=False)
    net.setInput(blob) 

    outputNames = net.getUnconnectedOutLayersNames() 
    
    # layerNames=net.getLayerNames()
    # #print(net.getUnconnectedOutLayers())
    # #print(layerNames)

    # outputNames=[layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)

    outputs=net.forward(outputNames)
    #print(len(outputs))

    findObjects(outputs,img)

 
    cv2.imshow('image',img)
    cv2.waitKey(1)

    
