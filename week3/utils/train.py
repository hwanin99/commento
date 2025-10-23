#### 예제 코드: YOLOv8을 활용한 객체 탐지 모델 학습
from ultralytics import YOLO

#YOLOv8 모델 로드
model = YOLO("yolov8m.pt")

# 사용자 데이터셋으로 학습
model.train(data='./data.yaml', cfg='./cfg.yaml', augment=True)