from flask import render_template, request, Blueprint
import cv2
import numpy as np

photo_blueprint = Blueprint('photo_blueprint', __name__)

@photo_blueprint.route('/')
def home():
    return render_template('photo.html')

@photo_blueprint.route('/detect', methods=['POST'])
def detect():
    # Lấy đường dẫn tệp ảnh từ yêu cầu POST
    image_file = request.files['image']
    image_path = 'static/' + image_file.filename
    image_file.save(image_path)

    # Đọc mô hình và cấu hình của YOLO
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    # Đọc các lớp nhãn từ tệp yolo.names
    classes = []
    with open("yolo.names", "r") as f:
        classes = f.read().splitlines()

    # Đọc ảnh đầu vào
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Chuẩn bị dữ liệu đầu vào cho mạng YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)

    # Lấy tên các lớp đầu ra từ mạng YOLO
    output_layers_names = net.getUnconnectedOutLayersNames()

    # Lan truyền thuận để lấy đầu ra từ các lớp đầu ra
    layerOutputs = net.forward(output_layers_names)

    # Khởi tạo danh sách các hộp giới hạn, độ tin cậy và mã lớp
    boxes = []
    confidences = []
    class_ids = []

    # Xử lý đầu ra từ các lớp đầu ra để lấy thông tin về hộp giới hạn, độ tin cậy và mã lớp
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Áp dụng Non-Maximum Suppression để loại bỏ các hộp che chắn
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Chuẩn bị font chữ và màu sắc ngẫu nhiên cho việc hiển thị nhãn
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    # Chuyển indexes từ tuple sang numpy array
    indexes = np.array(indexes)

    num_objects = len(indexes)
    cv2.putText(img, "Objects: {}".format(num_objects), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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
        color = label_colors.get(label, (255, 255, 255))  # Mặc định:Trắng

        # Vẽ hộp giới hạn
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # Tính toán kích thước của vùng chứa nhãn
        label_size, baseline = cv2.getTextSize(label + " " + detection_percentage, font, 1, 1)

        # Vẽ hình chữ nhật nền cho nhãn
        cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), color, cv2.FILLED)

        # Hiển thị nhãn lên ảnh
        cv2.putText(img, label + " " + detection_percentage, (x, y - 10), font, 1, (0, 0, 0), 2)

    # Thay đổi tỉ lệ hiển thị ảnh
    scale_factor = 1
    resized_img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    # Lưu ảnh kết quả
    result_path = 'static/result.jpg'
    cv2.imwrite(result_path, resized_img)

    return render_template('result.html', image_path=image_path, result_path=result_path)