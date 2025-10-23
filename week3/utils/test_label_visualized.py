import os
import glob
import cv2

# 이미지 및 라벨 경로
image_dir = 'your_test_image_paths'
label_dir = 'your_test_label_paths'

# 결과를 저장할 경로 (선택 사항)
save_dir = './predict'
os.makedirs(save_dir, exist_ok=True)

# 모든 이미지 경로 가져오기
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')),key=lambda x: int(x.split('/')[-1].split('.')[0]))

for img_path in image_paths:
    # 이미지 불러오기
    image = cv2.imread(img_path)
    h, w, _ = image.shape

    # 해당 이미지 이름에 맞는 라벨 파일 찾기
    base_name = os.path.basename(img_path).replace('.jpg', '.txt')
    label_path = os.path.join(label_dir, base_name)

    # 라벨 파일이 존재하지 않을 경우 스킵
    if not os.path.exists(label_path):
        print(f"[경고] {label_path} 없음 — 건너뜀")
        continue

    # YOLO 형식의 라벨 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # 형식이 맞지 않으면 무시
        cls, x_center, y_center, width, height = map(float, parts)

        # YOLO 좌표 → 픽셀 좌표로 변환
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # 좌상단, 우하단 좌표 계산
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # 바운딩박스 및 클래스 시각화
        color = (0, 255, 0)  # 초록색
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{int(cls)}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # 결과 시각화
    cv2.imshow("YOLO Object Detection", image)
    # cv2.waitKey(500)  # 0이면 키 입력까지 대기, 500은 0.5초 대기

    # 결과 저장 (선택)
    save_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, image)

print("✅ 모든 이미지 시각화 완료")