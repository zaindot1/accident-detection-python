import glob
import shutil
import sys

import cv2
import torch
from PIL import Image, ImageFile
from torch import nn
from torch import optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from flask import Flask,flash,request,redirect,url_for, render_template, Response
from flask_mysqldb import MySQL
import pyaudio
import time
from math import log10
import audioop
import os
from werkzeug.utils import secure_filename

from accident_detection_decibel.distance_estimator.inference import inference
from accident_detection_decibel.distance_estimator.visualizer import visualizer
from detect import detect_accident


app = Flask(__name__)

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'accident_detection'

mysql = MySQL(app)

UPLOAD_FOLDER = 'static/uploads/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
   return render_template('Home.html')

@app.route('/login')
def login():
   return render_template('Login.html')

@app.route('/verify',methods=('GET', 'POST'))
def verify():
   if request.method == 'POST':
      email = request.form['email']
      password = request.form['password']
      print(email)
      cursor = mysql.connection.cursor()
      query_string = "SELECT * FROM users WHERE email = %s and password=%s"
      temp=cursor.execute(query_string, (email,password))
      if(temp==1):
         print(temp)
         return redirect(url_for('services'))

   return render_template('Login.html')


@app.route('/services')
def services():
   return render_template('Services.html')

@app.route('/services/accident-detection')
def accident_detection():
   return render_template('Accident-Detection.html')

@app.route('/live-streaming')
def live_streaming():
   return render_template('Live-Streaming.html')

@app.route('/live-streaming', methods=['POST'])
def upload_video():
   if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
   file = request.files['file']
   if file.filename == '':
      flash('No video selected for uploading')
      return redirect(request.url)
   else:
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      # print('upload_video filename: ' + filename)
      flash('Video successfully uploaded')
      videopath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


      detect_accident(filename)
      import datetime
      import calendar

      img_array=[]

      dirFiles=glob.glob('./results/frames/*.png')
      dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
      for filename in dirFiles:
         img = cv2.imread(filename)
         height, width, layers = img.shape
         size = (width, height)
         img_array.append(img)

      date = datetime.datetime.utcnow()
      utc_time = calendar.timegm(date.utctimetuple())
      video_name = str(utc_time) + '.mp4'

      out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

      for i in range(len(img_array)):
         out.write(img_array[i])
      out.release()

      shutil.move(video_name,'../accident_prediction')

      ImageFile.LOAD_TRUNCATED_IMAGES = True

      test_transforms = transforms.Compose([transforms.Resize(255),
                                            #  transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            ])

      model = models.densenet161()

      model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Linear(1000, 2),
                                       nn.LogSoftmax(dim=1))

      criterion = nn.NLLLoss()
      # Only train the classifier parameters, feature parameters are frozen
      optimizer = optim.Adam(model.parameters(), lr=0.001)
      scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

      model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
      classes = ["accident", "noaccident"]
      count = 0
      counts = 1
      videopath = os.path.join(app.config['UPLOAD_FOLDER'], video_name)


      img_array = []

      vid = cv2.VideoCapture(videopath)
      ret = True
      while ret:
         if ret == True:
            ret, frame = vid.read()

            try:
               img = Image.fromarray(frame)
            except ValueError:
               break
            except AttributeError:
               break
            img = test_transforms(img)
            img = img.unsqueeze(dim=0)
            model.eval()
            with torch.no_grad():
               output = model(img)
               _, predicted = torch.max(output, 1)

               index = int(predicted.item())
               if index == 0:
                  count += 1
                  if counts == 1:
                     counts += 1

               labels = 'status: ' + classes[index]
            cv2.putText(frame, labels, (10, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
            img_array.append(frame)
            # print(frame)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

      vid.release()
      cv2.destroyAllWindows()

      path = './static/predicted_images'
      for i in range(len(img_array)):
         img = str(i) + '.jpg'
         cv2.imwrite(os.path.join(path, img), img_array[i])

      img_array = []

      dirFiles = glob.glob('./static/predicted_images/*.jpg')
      dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
      for filename in dirFiles:
         img = cv2.imread(filename)
         height, width, layers = img.shape
         size = (width, height)
         img_array.append(img)


      out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

      for i in range(len(img_array)):
         out.write(img_array[i])
      out.release()

      shutil.move(video_name, './accident_prediction')








      return render_template('Accident-Detection.html')
@app.route('/services/decibel-measurement')
def decibel_measurement():
   return render_template('Decibel-Detection.html')
rms=0
@app.route('/services/decibel-measurement/measure')
def decibel_measure():
   return render_template('Decibel-Detection-Measurement.html')
@app.route('/services/decibel-measurement/measure',methods=['POST'])
def decibel_measurement_video():
   if 'file' not in request.files:
      flash('No file part')
      return redirect(request.url)
   file = request.files['file']
   if file.filename == '':
      flash('No video selected for uploading')
      return redirect(request.url)
   else:
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      # print('upload_video filename: ' + filename)
      flash('Video successfully uploaded')
      videopath = os.path.join(app.config['UPLOAD_FOLDER'], filename)


      detect_accident(filename)
      inference()
      visualizer()

      import datetime
      import calendar

      img_array = []

      dirFiles = glob.glob('../object_detector/results/frames/*.png')
      dirFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

      for filename in dirFiles:
         img = cv2.imread(filename)
         height, width, layers = img.shape
         size = (width, height)
         print(size)
         img_array.append(img)

      date = datetime.datetime.utcnow()
      utc_time = calendar.timegm(date.utctimetuple())
      video_name = str(utc_time) + '.mp4'

      out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

      for i in range(len(img_array)):
         out.write(img_array[i])
      out.release()

      shutil.move(video_name, '../decibel_prediction')



   p = pyaudio.PyAudio()
   WIDTH = 2
   RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
   DEVICE = p.get_default_input_device_info()['index']
   print(p.get_default_input_device_info())

   def callback(in_data, frame_count, time_info, status):
      ref = 1
      global rms
      rms = audioop.rms(in_data, WIDTH) / ref
      return in_data, pyaudio.paContinue

   stream = p.open(format=p.get_format_from_width(WIDTH),
                   input_device_index=DEVICE,
                   channels=1,
                   rate=RATE,
                   input=True,
                   output=False,
                   stream_callback=callback)
   stream.start_stream()
   start_time = time.time()
   seconds = 10
   data = []
   while stream.is_active():
      if (rms > 0):
         db = 20 * log10(rms)
         if (db >= 0):
            print(f"RMS: {rms} DB: {db}")
            data.append(db)
      current_time = time.time()
      elapsed_time = current_time - start_time

      if elapsed_time > seconds:
         print("Finished iterating in: " + str(int(elapsed_time)) + " seconds")
         break
      time.sleep(0.3)
   stream.stop_stream()
   stream.close()

   p.terminate()



@app.route('/services/report')
def view_report():
   return render_template('Report.html')

@app.route('/services/accident-detection/view-video')
def view_video():
   ImageFile.LOAD_TRUNCATED_IMAGES = True

   test_transforms = transforms.Compose([transforms.Resize(255),
                                         #  transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         ])

   model = models.densenet161()

   model.classifier = nn.Sequential(nn.Linear(2208, 1000),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(1000, 2),
                                    nn.LogSoftmax(dim=1))

   criterion = nn.NLLLoss()
   # Only train the classifier parameters, feature parameters are frozen
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

   model.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
   classes = ["accident", "noaccident"]
   count = 0
   counts = 1
   videopath = '12.mp4'

   vid = cv2.VideoCapture(videopath)
   ret = True
   while ret:
      if ret == True:
         ret, frame = vid.read()

         try:
            img = Image.fromarray(frame)
         except ValueError:
            break
         except AttributeError:
            break
         img = test_transforms(img)
         img = img.unsqueeze(dim=0)
         model.eval()
         with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)

            index = int(predicted.item())
            if index == 0:
               count += 1
               if counts == 1:
                  counts += 1

            labels = 'status: ' + classes[index]

         cv2.putText(frame, labels, (10, 100),
                     cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5, cv2.LINE_AA)
         cv2.imshow('Frame', frame)
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break

   vid.release()
   cv2.destroyAllWindows()
   return render_template('View-Video.html')




if __name__ == '__main__':
   app.run(host="0.0.0.0",debug=True)