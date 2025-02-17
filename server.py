from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# ===============================
# YOLO 모델 파일 경로 설정
# ===============================
YOLO_WEIGHTS = "./AI/CRNN/od/yolov4_nia_od.weights"
YOLO_CFG = "./AI/CRNN/od/yolov4_nia_od.cfg"
YOLO_CLASSES = "./AI/CRNN/od/yolov4_nia_od.names"

# CRNN에서 출력할 Output 카테고리 리스트 (OCR 가능한 문자)
categories = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "_"]

# OCR 예측된 숫자 인덱스를 문자로 변환하는 StringLookup 생성
num_to_char = keras.layers.StringLookup(
    vocabulary=categories, num_oov_indices=1, mask_token=None, invert=True
)

# ===============================
# CRNN 모델 로드 및 입력 문제 해결
# ===============================
crnn_model = tf.keras.models.load_model("./AI/CRNN/20220104-1")

# 예측용 모델 (label 입력 제거)
prediction_model = keras.models.Model(
    inputs=crnn_model.get_layer(name="image").input,
    outputs=crnn_model.get_layer(name="dense2").output
)

# ===============================
# YOLO 모델 로드
# ===============================
with open(YOLO_CLASSES, "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CFG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# ===============================
# OCR 결과 디코딩 함수 (CTC Decode)
# ===============================
def decode_predictions(preds):
    """CRNN 모델의 출력을 문자열로 변환"""
    input_len = np.ones(preds.shape[0]) * preds.shape[1]
    results = keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]

    decoded_texts = []
    for res in results:
        res = tf.strings.reduce_join([num_to_char(x) for x in res if x != -1]).numpy().decode("utf-8")
        res = res.replace("[UNK]", "").replace("_", "")
        decoded_texts.append(res)

    return decoded_texts

# ===============================
# OCR 수행 함수 (이미지 개선 포함)
# ===============================
def ocr_with_crnn(image):
    """이미지를 CRNN OCR 모델에 입력하여 텍스트를 예측하는 함수"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    image = cv2.resize(image, (75, 375))  # ✅ 모델이 기대하는 크기로 변환
    image = image / 255.0  # 정규화
    image = np.expand_dims(image, axis=-1)  # 채널 차원 추가 (H, W, C)
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가 (1, H, W, C)

    preds = prediction_model.predict(image)
    decoded_texts = decode_predictions(preds)
    
    return decoded_texts[0]  # 최종 OCR 문자열 반환

# ===============================
# YOLO를 이용한 객체 탐지 및 OCR 수행
# ===============================
def detect_and_ocr(image_path):
    result = []
    """YOLO를 이용해 객체를 탐지하고, CRNN을 통해 OCR을 수행하는 함수"""
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 이미지를 찾을 수 없습니다. 경로를 확인하세요.")
        return

    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    detections = net.forward(output_layers)

    detected_boxes = []
    
    # YOLO 검출 결과 처리
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in ["BIC", "TYPESIZE"]:
                center_x, center_y, w, h = map(int, obj[:4] * [width, height, width, height])
                x, y = center_x - w // 2, center_y - h // 2
                
                x, y = max(0, x), max(0, y)
                w, h = min(width - x, w), min(height - y, h)
                
                detected_boxes.append((x, y, w, h))

    # OCR 수행
    for idx, (x, y, w, h) in enumerate(detected_boxes):
        roi = image[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        ocr_result = ocr_with_crnn(roi)  # ✅ idx 제거
        result.append(ocr_result)
        print(f"OCR 결과 [{idx}]: {ocr_result}")
    
    return result


app = Flask(__name__)

# 📌 📸 OCR 실행 API
@app.route("/predictLogistics", methods=["POST"])
def predict_logistics():
    data = request.get_json()
    image_path = data.get("image_path")

    if not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "이미지 경로가 올바르지 않습니다."}), 400

    ocr_result = detect_and_ocr(image_path)
    print("결과", ocr_result)
    return jsonify({"ocr_results": ocr_result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=9000, debug=True)