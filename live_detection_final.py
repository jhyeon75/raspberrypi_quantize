import numpy as np
import cv2
import time
# Picamera2 라이브러리 임포트 (CSI 카메라 사용을 위해)
from picamera2 import Picamera2
# tflite_runtime 라이브러리 임포트
from tflite_runtime.interpreter import Interpreter


# --- 1. 설정 변수 ---
# TFLite 모델 및 학습 시 설정값을 여기에 맞게 조정
MODEL_PATH = '/home/booxbox/yolov5/best-fp16.tflite'
INPUT_SIZE = 480          # 학습 시 사용한 이미지 크기 (480x480)
CONF_THRESHOLD = 0.25     # 객체 신뢰도 임계값
IOU_THRESHOLD = 0.45      # NMS IoU 임계값
CLASSES = ['With Helmet', 'Without Helmet', 'licence'] # data.yaml의 names 순서대로

# 모델 로드 및 초기화는 프로그램 시작 시 한 번만 수행
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# --- 2. 후처리 함수 (NMS 및 시각화) ---
# TFLite 출력 결과(output_raw)를 원본 이미지(img_original)에 그리는 역할
def post_process(output_raw, img_original, h_orig, w_orig):
    predictions = output_raw[0]
    boxes = []
    confidences = []
    class_ids = []

    # 2-A. 신뢰도 필터링 및 박스 좌표 변환
    for pred in predictions:
        score = pred[4] # 객체 신뢰도
        if score >= CONF_THRESHOLD:
            # 클래스 확률 중 가장 높은 값 찾기
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            confidence = score * class_scores[class_id] # 최종 신뢰도

            # YOLOv5 출력 좌표 [center_x, center_y, width, height] (0-1 정규화)
            center_x, center_y, w, h = pred[0:4]

            # 픽셀 좌표로 변환 (좌측 상단 x, y)
            x = int((center_x - w / 2) * w_orig)
            y = int((center_y - h / 2) * h_orig)
            w = int(w * w_orig)
            h = int(h * h_orig)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

    # 2-B. NMS (Non-Maximum Suppression) 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)

    # 2-C. 시각화 (결과 그리기)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y, w, h) = boxes[i]
            
            # 헬멧/미착용에 따라 색상 변경 가능
            color = (0, 255, 0) if CLASSES[class_ids[i]] == 'helmet' else (0, 0, 255)
            
            # 바운딩 박스 그리기
            cv2.rectangle(img_original, (x, y), (x + w, y + h), color, 2)
            
            # 라벨 및 신뢰도 표시
            label = f"{CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(img_original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    return img_original


# --- 3. 실시간 비디오 처리 및 추론 메인 루프 ---
def live_inference_picam2():
    # Picamera2 초기화 및 설정
    picam2 = Picamera2()
    # 캡처 크기 설정 (예: 모니터에 표시할 크기)
    # TFLite 입력 크기와 다르게 설정하여 디스플레이 해상도를 높일 수 있습니다.
    DISPLAY_W, DISPLAY_H = 640, 480 
    
    # 카메라 설정: main: 디스플레이/후처리용, raw: TFLite 입력용으로 사용 가능하지만,
    # 여기서는 main을 INPUT_SIZE에 맞추고 후처리 시 원본 크기로 스케일링하는 방식을 사용합니다.
    # 하지만 프레임 드랍을 줄이기 위해 한 가지 스트림만 사용하여 처리하는 것이 좋습니다.
    config = picam2.create_video_configuration(main={"size": (DISPLAY_W, DISPLAY_H), "format": "BGR888"}) 
    picam2.configure(config)
    picam2.start()

    print("✅ Picamera2 실시간 객체 탐지 시작... (종료: 'q' 키 입력)")
    
    try:
        while True:
            # 1. 프레임 캡처 (BGR888 형식으로 캡처하여 OpenCV와 호환)
            frame_original = picam2.capture_array() 

            h_orig, w_orig, _ = frame_original.shape

            # 2. 이미지 전처리 (TFLite 입력 크기 480x480으로 변환)
            img_resized = cv2.resize(frame_original, (INPUT_SIZE, INPUT_SIZE))
            input_data = (img_resized.astype(np.float32) / 255.0)
            input_data = np.expand_dims(input_data, axis=0)

            # 3. 추론 실행
            interpreter.set_tensor(input_details[0]['index'], input_data)
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            
            output_raw = interpreter.get_tensor(output_details[0]['index'])

            # 4. 후처리 및 시각화
            # 원본 크기(640x480) 프레임에 결과 박스를 그립니다.
            frame_output = post_process(output_raw, frame_original, h_orig, w_orig)

            # 5. FPS 표시
            fps = 1 / inference_time
            cv2.putText(frame_output, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 6. 화면 출력
            cv2.imshow('YOLOv5 TFLite Live Detection', frame_output)
            
            # 'q' 키를 누르거나 ESC를 누르면 종료
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27: 
                break

    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    finally:
        # 종료 시 리소스 해제
        picam2.stop()
        cv2.destroyAllWindows()


# --- 4. 실행 ---
if __name__ == '__main__':
    live_inference_picam2()