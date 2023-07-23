from flask import Flask, render_template, Response, request,flash
import cv2
import datetime, time
import os, sys
import numpy as np
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.patches as patches


app = Flask(__name__)
app.config['SECRET_KEY']='HELLO'


global face,capture
capture=0
face=0

#make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('saved templates/saved models/deploy.prototxt.txt', 'saved templates/saved models/res10_300x300_ssd_iter_140000.caffemodel')

camera = cv2.VideoCapture(0)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame
 

def gen_frames():  # generate frame by frame from camera
    global capture,face 
    while True:
        success, frame = camera.read() 
        if success:
         
            if(face):            
                frame= detect_face(frame)
            if(capture):
                capture=0
                p = os.path.sep.join(['shots', "shot.jpg"])
                cv2.imwrite(p, frame)
                
          
               
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


def getEmotion(file):
    detector = FER(mtcnn=True)
    img = plt.imread(file)
    result = detector.detect_emotions(img)
    if result:
        emotion, score = _getTopEmotion(result)
        return [emotion, score]
    else:
        return []


def _getTopEmotion(result):
    return sorted(list(result[0]["emotions"].items()), key=lambda x: x[1])[-1]


@app.route('/')
def home_page():
  return render_template('base.html')

@app.route('/capture_page')
def camera_page():
    return render_template('camera.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            flash("Your image has been captured successfully!!!!")
            global capture
            capture=1
            

    elif request.method=='GET':
        return render_template('camera.html')
    
    return render_template('camera.html')

@app.route('/emotion')
def emotion_detection():
   file = 'shots\shot.jpg'
   resultof =  getEmotion(file)
   emo_val = resultof[0]
   emotion_dict = { "happy"     : "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
              "sad"       : "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
              "angry"     : "https://open.spotify.com/playlist/5s7Sp5OZsw981I2OkQmyrz",
              "neutral"   : "https://open.spotify.com/playlist/37i9dQZF1E4p8lDdAdWt03",  
              "fearful"   : "https://open.spotify.com/playlist/7rzS9iLiqjy65AsZd9qinf",
              "disgusted" : "https://open.spotify.com/playlist/3qgzMg4m5tvf16PzlPgGa9",
              "surprised" : "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0"}
   
   return render_template ('emotion_detection.html',value = str(emo_val),linkval = emotion_dict[emo_val])



@app.route('/playist',methods=['POST','GET'])
def playlist(): 
    return render_template('playlist.html')


if __name__ == '__main__':
    app.run(debug=True)

camera.release() 
cv2.destroyAllWindows() 
     
     

