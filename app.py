
import base64
from io import BytesIO
import io
import requests
import cv2
from PIL import Image
from flask import Flask, request, jsonify,render_template
import numpy as np
import tensorflow.lite as tflite
import tensorflow as tf
from flask_cors import CORS, cross_origin
from flask_socketio import emit, SocketIO

app = Flask(__name__)
CORS(app,resources={r"/*":{"origins":"*"}})
socketio = SocketIO(app,cors_allowed_origins="*")

app.config['SECRET_KEY'] = 'secret!'
app.config['CORS_HEADERS'] = 'Content-Type'


interpreter=None
count=0
@socketio.on('frame')
def handle_frame(data):
  
  global interpreter
  image_src = data['imageSrc']
  screenshot_bytes = base64.b64decode(image_src.split(',')[1])
  img = Image.open(BytesIO(screenshot_bytes))
  opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
  print("type of image",type(img))
  img_rgb = cv2.resize(opencvImage, (300, 300), cv2.INTER_AREA)
  img_rgb = img_rgb.reshape([1, 300, 300, 3])
  
    ## *************************
  global count
  count=count+1
  print("Now, Count is",count)
  label_map = load_label_map('label_map.txt')
  input_details = interpreter.get_input_details()
  


    ##Loading model
  interpreter.set_tensor(input_details[0]['index'],   img_rgb)
  interpreter.invoke()
  output_details = interpreter.get_output_details()     
  output_dict = {
                 'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                 'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                 'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                 'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                 }
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  output_dict = apply_nms(output_dict,0.5, 0.6)
  results = []     
  for i in range(len(output_dict['detection_boxes'])):
       ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
       results.append({
          'class_name':label_map[output_dict['detection_classes'][i]],
          'score':float(output_dict['detection_scores'][i]),
          'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
      })
        
       
  socketio.emit('list_data', results)




@app.route('/detect', methods=['POST'])
def detect_objects():
    
    # Load the TensorFlow Lite model and label map
    global interpreter
    
  
    # # Process the incoming image
    imageData =  request.files.get("image")
    
    # img=Image.open(io.BytesIO(imageData.read()))
    img=Image.open(imageData)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    img_rgb = cv2.resize(opencvImage, (300, 300), cv2.INTER_AREA)
    img_rgb = img_rgb.reshape([1, 300, 300, 3])
  
    ## *************************

    label_map = load_label_map('label_map.txt')
    input_details = interpreter.get_input_details()
  


    ##Loading model
    interpreter.set_tensor(input_details[0]['index'],   img_rgb)
    interpreter.invoke()
    output_details = interpreter.get_output_details()

    output_dict = {
                   'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                   'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                   'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                   'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                   }
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    output_dict = apply_nms(output_dict,0.5, 0.6)
    results = []

    for i in range(len(output_dict['detection_boxes'])):
         ymin, xmin, ymax, xmax = output_dict['detection_boxes'][i]
         results.append({
            'class_name':label_map[output_dict['detection_classes'][i]],
            'score':float(output_dict['detection_scores'][i]),
            'bbox': [float(xmin), float(ymin), float(xmax), float(ymax)]
        })
        
       
    return jsonify(results)


        
def apply_nms(output_dict, iou_thresh=0.5, score_thresh=0.6):

    q = 90 # no of classes
    num = int(output_dict['num_detections'])
    boxes = np.zeros([1, num, q, 4])
    scores = np.zeros([1, num, q])
    # val = [0]*q
    for i in range(num):
        # indices = np.where(classes == output_dict['detection_classes'][i])[0][0]
        boxes[0, i, output_dict['detection_classes'][i], :] = output_dict['detection_boxes'][i]
        scores[0, i, output_dict['detection_classes'][i]] = output_dict['detection_scores'][i]
    nmsd = tf.image.combined_non_max_suppression(boxes=boxes,
                                                 scores=scores,
                                                 max_output_size_per_class=num,
                                                 max_total_size=num,
                                                 iou_threshold=iou_thresh,
                                                 score_threshold=score_thresh,
                                                 pad_per_class=False,
                                                 clip_boxes=False)
    valid = nmsd.valid_detections[0].numpy()
    output_dict = {
                   'detection_boxes' : nmsd.nmsed_boxes[0].numpy()[:valid],
                   'detection_classes' : nmsd.nmsed_classes[0].numpy().astype(np.int64)[:valid],
                   'detection_scores' : nmsd.nmsed_scores[0].numpy()[:valid],
                   }
    return output_dict

def load_label_map(path):
    with open(path) as f:
        lines = f.readlines()
    label_map = {}
    for line in lines:
        parts = line.strip().split(':')
        label_id = int(parts[0])
        label_name = parts[1]
        label_map[label_id] = label_name
    return label_map




if __name__ == '__main__':
    count=0
    interpreter = tflite.Interpreter(model_path='./detect.tflite')
    interpreter.allocate_tensors()
    socketio.run(app, debug=True,port=5001)

