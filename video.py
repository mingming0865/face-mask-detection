from flask import render_template, Response, request, Blueprint
import cv2
import numpy as np
from datetime import datetime
import os

video_blueprint = Blueprint('video_blueprint', __name__)

def generate_frames(video_path):
    start_time = datetime.now()
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open("yolo.names", "r") as f:
        classes = f.read().splitlines()
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))

    cap = cv2.VideoCapture(video_path)

    while True:
        elapsed_time = datetime.now() - start_time
        total_seconds = elapsed_time.total_seconds()
        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        seconds = int((total_seconds % 3600) % 60)

        _, img = cap.read()
        if img is None:
            break

        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)
        boxes = []
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        cv2.putText(img, "Total Time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        num_objects = len(indexes)
        cv2.putText(img, "Objects: {}".format(num_objects), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if len(indexes)>0:
            label_colors = {
                "none": (0, 0, 255),
                "bad": (0, 255, 255),
                "good": (0, 255, 0)
            }

            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                detection_percentage = str(round(confidences[i] * 100, 2)) + "%"
                color = label_colors.get(label, (255, 255, 255))
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                label_size, baseline = cv2.getTextSize(label + " " + detection_percentage, font, 2, 2)
                cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, cv2.FILLED)
                cv2.putText(img, label + " " + detection_percentage, (x, y - 10), font, 2, (0, 0, 0), 2)

        max_width = 1280
        max_height = 720
        scale_factor = min(max_width / width, max_height / height)
        resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        ret, buffer = cv2.imencode('.jpg', resized_img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@video_blueprint.route('/')
def index():
    return render_template('video.html')

@video_blueprint.route('/video_feed', methods=['POST'])
def video_feed():
    if 'video' not in request.files:
        return "No video uploaded", 400

    video_file = request.files['video']
    video_path = os.path.join('static', video_file.filename)
    video_file.save(video_path)

    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
