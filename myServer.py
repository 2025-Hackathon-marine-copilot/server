from myEnv import ShipEnv  # 환경 불러오기
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from flask import Flask, request, jsonify
import os
import numpy as np
import cv2

app = Flask(__name__)

# 🚀 학습된 DQN 모델 불러오기
model = DQN.load("route_30000",env = ShipEnv(),device="mps")

@app.route("/predict", methods=["POST"])
def predict():
    """ 📌 출발지(start)와 목적지(goal)를 받아 최적 경로를 반환하는 API """
    
    data = request.get_json()
    start = tuple(data.get("start", (0, 0)))
    goal = tuple(data.get("goal", (22, 14)))
    moving = list(data.get("movingShip",[]))
    print(moving)
    anchored = tuple(data.get("anchoredShip",[]))
    print(f"🚀 요청: start={start}, goal={goal}")
    settlement_vessel = [(14,13),(16,13),(11, 6)] + moving

    # 🌊 환경 초기화
    env = DummyVecEnv([lambda: Monitor(ShipEnv(start_pos=start, goal_pos=goal, settlement_vessel=settlement_vessel))])
    obs = env.reset()
    done = False
    ship_path = []

    # 🚢 탐험 비활성화
    model.exploration_initial_eps = 0.0
    model.exploration_final_eps = 0.0

    # 🌊 시뮬레이션 실행
    while not done:
        action, _ = model.predict(obs, deterministic=True)  # ✅ 탐험 비활성화
        obs, reward, terminated, truncated = env.step(action)
        x, y = int(obs[0][-2]), int(obs[0][-1])
        ship_path.append((x, y))
        done = terminated[0]

    print(f"✅ 최적 경로 반환: {ship_path}")

    # 📌 결과 반환
    return jsonify({"path": ship_path, "length": len(ship_path)})

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
    app.run(host="0.0.0.0", port=9200, debug=True)