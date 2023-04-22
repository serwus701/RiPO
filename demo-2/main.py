# import tensorflow_hub as hub
# import cv2
# import numpy
# import tensorflow as tf
# import pandas as pd

# # Carregar modelos
# detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")
# labels = pd.read_csv('labels.csv',sep=';',index_col='ID')
# labels = labels['OBJECT (2017 REL.)']

# cap = cv2.VideoCapture(0)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# while(True):
#     #Capture frame-by-frame
#     ret, rgb = cap.read()

#     #Is optional but i recommend (float convertion and convert img to tensor image)
#     rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

#     #Add dims to rgb_tensor
#     rgb_tensor = tf.expand_dims(rgb_tensor , 0)
    
#     boxes, scores, classes, num_detections = detector(rgb_tensor)
    
#     pred_labels = classes.numpy().astype('int')[0]
    
#     pred_labels = [labels[i] for i in pred_labels]
#     pred_boxes = boxes.numpy()[0].astype('int')
#     pred_scores = scores.numpy()[0]
   
#    #loop throughout the detections and place a box around it  
#     for score, (ymin,xmin,ymax,xmax), label in zip(pred_scores, pred_boxes, pred_labels):
#         if score < 0.5:
#             continue
            
#         score_txt = f'{score:.2f}'
#         img_boxes = cv2.rectangle(rgb,(xmin, ymax),(xmax, ymin),(0,255,0),1)      
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         cv2.putText(img_boxes,label,(xmin, ymin - 10), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
#         cv2.putText(img_boxes,score_txt,(xmax - 10, ymin - 10), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
        

#     #Display the resulting frame
#     cv2.imshow('black and white', rgb)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import tensorflow_hub as hub
import cv2
import tensorflow as tf
import env_extractor


# Define constants
MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1"
LABELS_PATH = 'labels.csv'
LABELS_COLUMN = 'OBJECT (2017 REL.)'
CAP_PROP_FRAME_WIDTH = 1920
CAP_PROP_FRAME_HEIGHT = 1080
FONT = cv2.FONT_HERSHEY_SIMPLEX
SCORE_THRESHOLD = 0.5

def setup():
    # Load models
    detector, labels = env_extractor.load_env_variables(MODEL_URL, LABELS_PATH, LABELS_COLUMN)

    # Set up video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_HEIGHT)

    return detector, labels, cap

def to_tensor(img):
     # Convert image to tensor
    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.uint8)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor , 0)

    return rgb_tensor

def draw_boxes(frame, boxes, scores, labels):
    # Loop through detections and draw bounding boxes on image
    for score, (ymin,xmin,ymax,xmax), label in zip(scores, boxes, labels):
        if score < SCORE_THRESHOLD:
            continue

        score_txt = f'{score:.2f}'
        img_boxes = cv2.rectangle(frame,(xmin, ymax),(xmax, ymin),(0,255,0),1)
        cv2.putText(img_boxes,label,(xmin, ymin - 10), FONT, 0.5, (0,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_boxes,score_txt,(xmax - 10, ymin - 10), FONT, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return frame

def predict(detector, tensor_img, labels):
    # Run object detection model
    boxes, scores, classes, num_detections = detector(tensor_img)

    # Convert predictions to numpy arrays and extract labels
    pred_labels = classes.numpy().astype('int')[0]
    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    return pred_boxes, pred_scores, pred_labels

def main():
    detector, labels, cap = setup()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        tensor_img = to_tensor(frame)

        pred_boxes, pred_scores, pred_labels = predict(detector, tensor_img, labels)

        draw_boxes(frame, pred_boxes, pred_scores, pred_labels)

        # Display the resulting frame
        cv2.imshow('live capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
