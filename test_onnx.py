import onnxruntime as ort
import numpy as np
import cv2 # OpenCV 라이브러리

def postprocess(output_data, conf_threshold, original_image):
    """
    YOLOv8 ONNX 모델의 출력을 후처리하여 바운딩 박스, 신뢰도, 클래스 ID를 추출하는 함수
    """
    # 1. 출력 데이터 형태 변환: (1, 12, 8400) -> (1, 8400, 12)
    output_data = np.transpose(output_data, (0, 2, 1))
    
    # 2. 이미지 크기에 맞춰 박스 좌표 스케일링 준비
    img_height, img_width = original_image.shape[:2]
    model_height, model_width = 320, 320 # 모델 입력 크기
    x_scale = img_width / model_width
    y_scale = img_height / model_height
    
    detections = []
    
    # 3. 8400개의 예측 결과에 대해 반복
    for pred in output_data[0]:
        # 박스 좌표(cx, cy, w, h)와 클래스 점수 추출
        box = pred[:4]
        class_scores = pred[4:]
        
        # 4. 가장 높은 클래스 점수와 해당 클래스 ID 찾기
        class_id = np.argmax(class_scores)
        max_score = class_scores[class_id]
        
        # 5. 신뢰도 임계값(conf_threshold)을 넘는 예측만 처리
        if max_score > conf_threshold:
            # 중심점(cx, cy)과 너비/높이(w, h) 추출
            cx, cy, w, h = box
            
            # 원본 이미지 크기에 맞게 좌표 변환
            x1 = int((cx - w / 2) * x_scale)
            y1 = int((cy - h / 2) * y_scale)
            x2 = int((cx + w / 2) * x_scale)
            y2 = int((cy + h / 2) * y_scale)
            
            detections.append({
                "class_id": class_id,
                "score": float(max_score),
                "box": [x1, y1, x2, y2]
            })

    # (선택) NMS (Non-Maximum Suppression)를 여기에 적용하면 더 좋습니다.
    # OpenCV의 dnn.NMSBoxes 함수를 사용하거나 직접 구현할 수 있습니다.
            
    return detections

def draw_boxes(image, detections):
    """탐지된 객체들의 바운딩 박스를 이미지에 그리는 함수"""
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        score = det['score']
        
        x1, y1, x2, y2 = box
        
        # 박스 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 텍스트 추가
        label = f"Class {class_id}: {score:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return image

def detect_objects_from_array(session, image_array, conf_threshold=0.5):
    """
    NumPy 배열 형태의 이미지 데이터를 받아 객체 탐지를 수행하고 결과를 반환합니다.

    Args:
        session (onnxruntime.InferenceSession): 로드된 ONNX 모델 세션
        image_array (np.ndarray): 분석할 이미지 데이터 (OpenCV BGR 형식)
        conf_threshold (float): 신뢰도 임계값

    Returns:
        list: 탐지된 객체 정보(딕셔너리)들을 담고 있는 리스트.
    """
    input_name = session.get_inputs()[0].name
    
    # 1. 이미지 전처리 (파일 읽기 과정이 image_array를 직접 사용하는 것으로 변경됨)
    original_image = image_array
    
    # 이미지 리사이즈, 정규화, 차원 변경
    input_img = cv2.resize(original_image, (320, 320))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_tensor = np.expand_dims(input_img, axis=0)
    
    # 2. 추론 실행
    result = session.run(None, {input_name: input_tensor})
    
    # 3. 결과 후처리
    output_data = result[0]
    detections = postprocess(output_data, conf_threshold, original_image)
    
    # 4. 최종 결과 반환
    return detections

def main():
    # --- 설정 ---
    onnx_model_path = "./result/best5/best.onnx"
    image_to_test_path = "test_image.jpg"
    conf_threshold = 0.5 # 신뢰도 임계값 (0.5 이상인 것만 탐지)

    # 1. ONNX 세션 생성
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    
    # 2. 이미지 로드 및 전처리
    original_image = cv2.imread(image_to_test_path)
    # 이미지 리사이즈, 정규화, 차원 변경 (이전 preprocess 함수와 유사)
    input_img = cv2.resize(original_image, (320, 320))
    input_img = input_img.astype(np.float32) / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))
    input_tensor = np.expand_dims(input_img, axis=0)
    
    # 3. 추론 실행
    print("추론을 시작합니다...")
    result = session.run(None, {input_name: input_tensor})
    
    # 4. 결과 후처리
    output_data = result[0]
    detections = postprocess(output_data, conf_threshold, original_image)
    
    print(f"\n--- 탐지된 객체 수: {len(detections)} ---")
    for i, det in enumerate(detections):
        print(f"객체 {i+1}: 클래스 ID={det['class_id']}, 신뢰도={det['score']:.2f}, 박스={det['box']}")

    # 5. 결과 시각화
    result_image = draw_boxes(original_image.copy(), detections)
    cv2.imwrite("result_image.jpg", result_image)
    print("\n결과 이미지가 'result_image.jpg'로 저장되었습니다.")

if __name__ == "__main__":
    main()
