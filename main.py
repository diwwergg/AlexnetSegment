import cv2
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

# Path: main.py
# ENVIRONMENT VARIABLES
# Link download model: https://drive.google.com/drive/folders/1kxc-DJm_LqWTSXLyvy10pXUlfbwEeKbm?usp=share_link
CAMERA_NUMBER = 0
MODEL = 'Model/alexsegment.h5'
WIDTH = 640
HEIGHT = 480
ZONE_SIZE = 400

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU Divice: ", tf.test.gpu_device_name())


# cap = cv2.VideoCapture('videos/1.mp4')
cap = cv2.VideoCapture(CAMERA_NUMBER)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
w = (WIDTH-ZONE_SIZE)//2
h = (HEIGHT-ZONE_SIZE)//2
zone = [(w,h ), (w+ZONE_SIZE, h+ZONE_SIZE)]
green_color = (0, 255, 0)

# Load model
model = keras.models.load_model(MODEL)

def labelshow(one_hot):
  one_hot = np.array(one_hot).flatten()
  labels = np.array(['phone', 'ezygo', 'wallet', 'cara', 'harddisk'])
  onehots = np.array([[0, 0, 0, 0, 1]
            ,[0, 0, 0, 1, 0]
            ,[0, 0, 1, 0, 0]
            ,[0, 1, 0, 0, 0]
            ,[1, 0, 0, 0, 0]])
  # Find the index of the matching one-hot encoding in onehots
  index_arr = np.where((onehots == one_hot).all(axis=1))[0]
  if index_arr.size > 0:
    index = index_arr[0]
    return str(labels[index])
  return 'None'



def predict_by_image(image):
    img = np.array(cv2.resize(image, [256, 256]))
    (z1, z2) = model.predict(img[None, :, :, :], verbose = 0)
    # z1 is one hot array have 5 classes
    label = labelshow(z1[0].round().astype(int))
    if label == 'None':
        return (None, None, None)
    xyxy = np.array(z2*ZONE_SIZE).astype(int).flatten()
    return(np.array(z1), xyxy, label)
    

while True:
    t1 = time.time()
    ret, frame = cap.read()
    if ret == False:
        print("Error: Camera not found")
        break
    f1 = cv2.rectangle(frame.copy(), zone[0], zone[1], green_color, 2)
    image = frame.copy()[h:h+ZONE_SIZE,w:w+ZONE_SIZE]
    (z1, z2, label) = predict_by_image(image)
    if label != None:
        print(label, z2)
        [x1, y1, x2, y2] = z2
        (x3, y3) = (abs(x2-x1)//2, abs(y2-y1)//2 )
        cv2.rectangle(image, (x1, y1), (x2, y2), green_color, 1)
        cv2.putText(image, label, (x3, y3), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, green_color, 1)
    t2 = time.time()
    fps = round(1/(t2-t1),2)
    cv2.putText(f1, 'FPS: '+str(fps), (20, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.imshow('Frame1', f1)
    cv2.imshow('Frame2', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()