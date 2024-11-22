from flask import Flask, render_template, Response, Blueprint
import cv2
import numpy as np
import time
from datetime import datetime

webcam_blueprint = Blueprint('webcam_blueprint', __name__)

def detect_objects():
    start_time = datetime.now()

    # Đọc mô hình YOLO từ file
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # Đọc tên các lớp từ file
    classes = []
    with open("yolo.names", "r") as f:
        classes = f.read().splitlines()

    # Mở webcam để xử lý video trực tiếp
    cap = cv2.VideoCapture(0)

    while True:
        elapsed_time = datetime.now() - start_time
        
        total_seconds = elapsed_time.total_seconds()

        hours = int(total_seconds / 3600)
        minutes = int((total_seconds % 3600) / 60)
        seconds = int((total_seconds % 3600) % 60)

        _, img = cap.read()
        height, width, _ = img.shape

        # Chuẩn bị input cho mô hình YOLO
        blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)

        # Lấy output từ các lớp cuối cùng của mô hình
        output_layers_names = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_names)

        boxes = []
        confidences = []
        class_ids = []

        # Xử lý output và lấy thông tin về bounding box, confidence và class ID
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0]*width)
                    center_y = int(detection[1]*height)
                    w = int(detection[2]*width)
                    h = int(detection[3]*height)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        # Áp dụng non-maximum suppression để loại bỏ các bounding box trùng lắp
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        indexes = np.array(indexes)  # Chuyển tuple thành mảng numpy

        # Định nghĩa màu cho từng nhãn
        label_colors = {
            "none": (0, 0, 255),    # Đỏ
            "bad": (0, 255, 255),   # Vàng
            "good": (0, 255, 0)     # Xanh lá
        }

        # Vẽ hộp giới hạn và hiển thị nhãn trên ảnh đầu vào
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            detection_percentage = str(round(confidences[i] * 100, 2)) + "%"

            # Lấy màu cho nhãn từ từ điển màu
            color = label_colors.get(label, (255, 255, 255))  # Mặc định: Trắng

            # Vẽ hộp giới hạn
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # Tính toán kích thước của vùng chứa nhãn
            label_size, baseline = cv2.getTextSize(label + ""  + detection_percentage, font, 2, 2)

            # Vẽ hình chữ nhật nền cho nhãn
            cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, cv2.FILLED)

            # Hiển thị nhãn lên ảnh
            cv2.putText(img, label + " " + detection_percentage, (x, y - 10), font, 2, (0, 0, 0), 2)

        # Lấy thời gianhiện tại
        current_time = time.strftime("%I:%M:%S %p")

        # Hiển thị thời gian lên ảnh
        cv2.putText(img, "Clock: {}".format(current_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(img, "Total Time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        num_objects = len(indexes)
        cv2.putText(img, "Objects: {}".format(num_objects), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        scale_factor = 1.5  # Phóng to video lên 1.5 lần
        resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        # Chuyển ảnh thành chuỗi byte để gửi cho trình duyệt
        _, img_encoded = cv2.imencode('.jpg', resized_img)
        frame = img_encoded.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Thoát khỏi vòng lặp khi nhấn phím "q"
        if cv2.waitKey(1) == ord("q"):
            break

    # Giải phóng bộ nhớ và đóng cửa sổ
    cap.release()
    cv2.destroyAllWindows()

@webcam_blueprint.route('/')
def index():
    return render_template('webcam.html')

def gen():
    return detect_objects()

@webcam_blueprint.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')