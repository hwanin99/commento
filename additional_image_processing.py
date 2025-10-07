import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms


''' 
함수명
--------------------------------------------------------------
<기본 문제>
1. resize(root, show)
2-(1). gray(root, show)
2-(2). norm(root, show)
3. blur(root, show)
4-(1) flip(root, show)
4-(2) rotate(root, show)
4-(3) color_aug(root, show)
--------------------------------------------------------------
<심화 문제>
1. del_dark(root, show)
2. del_small_object(root, ratio)
--------------------------------------------------------------
* 현재 경로(root path)에 sample_0.jpg ~ sample_4.jpg가 있어야 함.
* show: default=True, ratio: default=3
'''

# root = './'


''' <기본 문제> '''
# 1. 크기 조정(224x224)
def resize(root, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    # 1. 각 이미지의 평균 밝기 계산
    for img in img_paths:
        image = cv2.imread(img)
        resize_img = cv2.resize(image,(224,224)) # (224x224)로 resize

        if show:
            cv2.imshow('Resized Image', resize_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return resize_img

# 2. 색상 변환(Gray Scale & Normalize 적용)
## (1) Gray Scale
def gray(root,show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # BGR -> GRAY

        if show:
            cv2.imshow('Gray Scale', gray_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return gray_img

## (2) Normalize
def norm(root, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)

        # 채널 별 mean, std
        mean = np.mean(image, axis=(0, 1)) # H, W 축 기준으로 채널 별 평균
        std = np.std(image, axis=(0, 1)) # H, W 축 기준으로 채널 별 표준 편차
        
        # Normalize
        normalize_img = (image - mean) / std # 정규화

        if show:
            cv2.imshow('Gray Scale', normalize_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    return normalize_img

# 3. 노이즈 제거(Blur 필터 적용)
def blur(image, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)
        # Gaussian Blur
        blur_img = cv2.GaussianBlur(image, (21,21), sigmaX=0, sigmaY=0)
        
        if show:
            cv2.imshow('Gaussian Blur', blur_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return blur_img

# 4. 데이터 증강 (좌우 반전, 회전, 색상 변환)
## (1) 좌우 반전
def flip(root, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)
        t_img = torch.tensor(image).permute(2,0,1) # array -> tensor & (H,W,C) -> (C,H,W)

        # 좌우 반전
        flip = transforms.RandomHorizontalFlip(p=1.0) # p는 확률
        flip_img = flip(t_img).permute(1,2,0).numpy() # 시각화를 위해 다시 (C,H,W) -> (H,W,C) 이후에 array로 변환
        
        if show:
            cv2.imshow('Flipped Image', flip_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return flip_img

## (2) 회전
def rotate(root, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)
        t_img = torch.tensor(image).permute(2,0,1)

        # 회전
        rotate = transforms.RandomRotation(180) # 180은 회전 각도 범위(±180)
        rotate_img = rotate(t_img).permute(1,2,0).numpy()

        if show:
            cv2.imshow('Rotated Image', rotate_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return rotate_img

## (3) 색상 변경
def color_aug(root, show=True):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    for img in img_paths:
        image = cv2.imread(img)
        t_img = torch.tensor(image).permute(2,0,1)

        # 색상변환
        color_aug = transforms.ColorJitter(
            brightness=0.3,   # 밝기 변화(±30%)
            contrast=0.3,     # 대비 변화(±30%)
            saturation=0.3,   # 채도 변화(±30%)
            hue=0.1           # radian 색상 변화(±0.1)
        )

        color_aug_img = color_aug(t_img).permute(1,2,0).numpy()

        if show:
            cv2.imshow('Color aug Image', color_aug_img)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return color_aug_img



''' <심화 문제> '''
### Matplot으로 subplot 사용. ###
# 너무 어두운 이미지 제거(평균 밝기 기준)
def del_dark(root, show=True):
    # 이미지 경로 수집
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
    brightness_list = []

    # 각 이미지의 평균 밝기 계산
    for img in img_paths:
        image = cv2.imread(img)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness_list.append(np.mean(gray_image))

    # 밝기 딕셔너리 출력
    brightness_dict = {f'{i}번': float(round(b, 4)) for i, b in enumerate(brightness_list)}
    print(f'각 이미지 평균 밝기: {brightness_dict}')

    # 전체 평균 밝기
    mean_brightness = np.mean(brightness_list)
    print(f'전체 평균 밝기: {round(mean_brightness, 4)}')

    # 평균보다 어두운 이미지 제거
    filtered_paths = [p for p, b in zip(img_paths, brightness_list) if b >= mean_brightness]
    filtered_brightness = [b for b in brightness_list if b >= mean_brightness]
    print(f'남은 이미지: {[os.path.basename(f) for f in filtered_paths]}')

    # 시각화
    if show:
        # Original Images
        plt.figure(figsize=(20, 5))
        plt.suptitle('Original Images', fontsize=16)
        for i, (img, b) in enumerate(zip(img_paths, brightness_list)):
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(img_paths), i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'{os.path.basename(img)}\n{round(b, 2)}')
        plt.tight_layout()
        plt.show()

        # Filtered Images
        plt.figure(figsize=(20, 5))
        plt.suptitle(f'Filtered Images (≥ {round(mean_brightness, 2)})', fontsize=16)
        for i, (img, b) in enumerate(zip(filtered_paths, filtered_brightness)):
            image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(filtered_paths), i+1)
            plt.imshow(image)
            plt.axis('off')
            plt.title(f'{os.path.basename(img)}\n{round(b, 2)}')
        plt.tight_layout()
        plt.show()

    return filtered_paths


# 2. 객체 크기가 너무 작은 이미지 제거
'''
이미지에 클릭을 통해 좌표 설정 
* 객체가 여러 개인 경우: 객체 하나 좌표 설정이 끝나면 'n'을 누르고 다음 객체 좌표 설정.
* 각 이미지의 좌표 설정이 끝나면 'esc'를 눌러 다음 이미지로 진행.
'''
def del_small_object(root, ratio=3):
    img_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))

    COLOR = (0, 255, 0)
    COLOR_text = (0, 0, 255)
    THICKNESS = 2
    SCALE = 0.5

    all_objects = {}  # 모든 이미지 좌표 저장

    def mouse_handler(event, x, y, flags, param):
        point_list, image = param['point_list'], param['image']

        if event == cv2.EVENT_LBUTTONDOWN:
            point_list.append((x, y))
            print(f'Point {len(point_list)}: ', (x, y))

        # 시각화
        temp_img = image.copy()
        prev_point = None
        for point in point_list:
            cv2.circle(temp_img, point, 5, COLOR, cv2.FILLED)
            cv2.putText(temp_img, f"({point})", point, cv2.FONT_HERSHEY_SIMPLEX, SCALE, COLOR_text, 1)
            if prev_point:
                cv2.line(temp_img, prev_point, point, COLOR, THICKNESS, cv2.LINE_AA)
            prev_point = point

        if len(point_list) > 2:
            cv2.line(temp_img, point_list[-1], point_list[0], COLOR, THICKNESS, cv2.LINE_AA)

        cv2.imshow('img', temp_img)

    # 이미지 반복
    for image_path in img_paths:
        image = cv2.imread(image_path)
        height, width = image.shape[:2]
        print(f'\n{os.path.basename(image_path)} -> 해상도: {width}x{height}, 전체 넓이: {width*height} pixels')

        objects = []  # 한 이미지 안의 객체 좌표 저장
        point_list = []

        cv2.namedWindow('img')
        cv2.setMouseCallback('img', mouse_handler, param={'point_list': point_list, 'image': image})

        while True:
            cv2.imshow('img', image)
            key = cv2.waitKey(0) & 0xFF

            if key == ord('n'):  # n을 누르면, 현재 객체 저장 -> 새로운 객체 시작
                if len(point_list) >= 3:
                    objects.append(point_list.copy())
                    print(f'객체 저장: {point_list}')
                point_list.clear()

            elif key == 27:  # ESC → 이미지 종료
                if len(point_list) >= 3:
                    objects.append(point_list.copy())
                    print(f'마지막 객체 저장: {point_list}')
                break

        cv2.destroyAllWindows()
        all_objects[os.path.basename(image_path)] = objects

        # 각 객체 면적 출력 + 합계 계산
        total_area = 0
        for i, obj in enumerate(objects):
            contour = np.array(obj, dtype=np.int32)
            area = cv2.contourArea(contour)
            total_area += area
            print(f'객체 {i+1} 면적: {area} pixels')
        print(f'{os.path.basename(image_path)} 전체 객체 면적: {total_area} pixels')

    # 전체 객체 면적 합 < 전체 이미지 넓이 1/ratio 제외
    filtered_objects = {}
    for image_name, objects in all_objects.items():
        img = cv2.imread(os.path.join(root, image_name))
        height, width = img.shape[:2]
        total_image_area = width * height

        total_objects_area = sum(cv2.contourArea(np.array(obj, dtype=np.int32)) for obj in objects)

        if total_objects_area >= total_image_area / ratio:
            filtered_objects[image_name] = objects

    # 객체 크기가 작은 이미지 제거 후 결과 출력
    print(f'\n객체 면적이 이미지의 1/{ratio} 이하인 이미지 제거:')
    for k, v in filtered_objects.items():
        print(f'남은 이미지: {k} -> (객체 수: {len(v)})')


    return all_objects, filtered_objects

