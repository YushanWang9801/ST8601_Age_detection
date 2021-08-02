import numpy as np
import cv2
import os
import argparse
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import keras


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load Model
prototxtPath = os.path.sep.join(["./face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join(["./face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

ageNet = keras.models.load_model('age_model_final.h5')

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "60+"]

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    # print(frames)
    
    # load the input image and construct an input blob for the image
    (h, w) = frames.shape[:2]
    blob = cv2.dnn.blobFromImage(frames, 1.0, (300, 300),(104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):	
        confidence = detections[0, 0, i, 2]
        
        
        if confidence > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frames[startY:endY, startX:endX]
            face = cv2.resize(face, (227,227) )
            predicted_label = ageNet.predict_classes(face.reshape((-1,227,227,3)))
            predictions = ageNet.predict(face.reshape((-1,227,227,3)))


            text = AGE_BUCKETS[int(predicted_label)] + "{:.2f}%".format(max(predictions)[0]*1000)
        
            

            y = startY - 10 if startY - 10 > 10 else startY + 10


            #print(AGE_BUCKETS[int(predicted_label)])
            color = (36,255,12)
            cv2.putText(frames, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frames, (startX, startY), (endX, endY), color, 2)
            
    # Display the resulting frame
    cv2.imshow('Video', frames)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()