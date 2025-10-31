#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.utils import load_image
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QGridLayout, QVBoxLayout
from PyQt5.QtGui import QFont, QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PIL import Image
import numpy as np

# ------------------- 모델 경로 -------------------
checkpoint_path = "./instruct_pix2pix_model/"
# Turn it into a creature with a white belly and a red face and back.

# ------------------- 유틸 함수 -------------------
def pil2pixmap(image):
    image = image.convert("RGBA")
    data = np.array(image)
    h, w, c = data.shape
    bytes_per_line = c * w
    qimg = QImage(data.data, w, h, bytes_per_line, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)

# ------------------- Worker Thread -------------------
class GenerateThread(QThread):
    finished = pyqtSignal(Image.Image)
    progress = pyqtSignal(int)

    def __init__(self, img_name, prompt, pipeline, num_steps=50):
        super().__init__()
        self.img_name = img_name
        self.prompt = prompt
        self.pipeline = pipeline
        self.num_steps = num_steps

    def run(self):
        image = load_image(f'./image2sketch/dataset/sketch/{self.img_name}')

        # callback 함수로 생성 진행률 업데이트
        def progress_callback(step, timestep, latents):
            percent = int((step + 1) / self.num_steps * 100)
            self.progress.emit(percent)

        # 생성
        result = self.pipeline(
            self.prompt,
            image=image,
            num_inference_steps=self.num_steps,
            callback=progress_callback,
            callback_steps=1
        ).images[0]

        self.finished.emit(result)

# ------------------- PyQt5 GUI -------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generate Your Pokemon")
        self.resize(1200, 600)

        centralwidget = QWidget(self)
        self.setCentralWidget(centralwidget)

        main_layout = QVBoxLayout(centralwidget)
        font = QFont()
        font.setBold(True)
        font.setWeight(75)

        # ===== 상단 입력 영역 =====
        top_layout = QGridLayout()

        label_image = QLabel("Image")
        label_image.setFont(font)
        label_image.setAlignment(Qt.AlignCenter)
        label_image.setFixedSize(150, 35)
        top_layout.addWidget(label_image, 0, 0)

        label_prompt = QLabel("Prompt")
        label_prompt.setFont(font)
        label_prompt.setAlignment(Qt.AlignCenter)
        label_prompt.setFixedSize(150, 35)
        top_layout.addWidget(label_prompt, 0, 1)

        self.lineEdit_image = QLineEdit()
        self.lineEdit_image.setFixedSize(150, 35)
        top_layout.addWidget(self.lineEdit_image, 1, 0)

        self.lineEdit_prompt = QLineEdit()
        self.lineEdit_prompt.setFixedSize(850, 35)
        top_layout.addWidget(self.lineEdit_prompt, 1, 1)

        self.pushButton = QPushButton("Generate")
        self.pushButton.setFont(font)
        top_layout.addWidget(self.pushButton, 1, 2)
        self.pushButton.clicked.connect(self.generate_image)
        self.lineEdit_image.returnPressed.connect(self.pushButton.click) #엔터키로도 가능
        self.lineEdit_prompt.returnPressed.connect(self.pushButton.click)

        main_layout.addLayout(top_layout)

        # ===== 하단 이미지 영역 =====
        bottom_layout = QGridLayout()

        label_original = QLabel("Original Pokemon")
        label_original.setFont(font)
        label_original.setAlignment(Qt.AlignCenter)
        label_original.setFixedSize(300, 35)
        bottom_layout.addWidget(label_original, 0, 0)

        label_generated = QLabel("Your Pokemon")
        label_generated.setFont(font)
        label_generated.setAlignment(Qt.AlignCenter)
        label_generated.setFixedSize(300, 35)
        bottom_layout.addWidget(label_generated, 0, 1)

        # Original 이미지 프레임
        self.label_original_img = QLabel()
        self.label_original_img.setFixedSize(300, 300)
        self.label_original_img.setAlignment(Qt.AlignCenter)
        self.label_original_img.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.label_original_img.setLineWidth(2)
        self.label_original_img.setText("Original Pokemon")
        bottom_layout.addWidget(self.label_original_img, 1, 0)

        # Generated 이미지 프레임
        self.label_generated_img = QLabel()
        self.label_generated_img.setFixedSize(300, 300)
        self.label_generated_img.setAlignment(Qt.AlignCenter)
        self.label_generated_img.setFrameStyle(QLabel.Box | QLabel.Plain)
        self.label_generated_img.setLineWidth(2)
        self.label_generated_img.setText("Waiting...")
        bottom_layout.addWidget(self.label_generated_img, 1, 1)

        main_layout.addLayout(bottom_layout)

        # Stable Diffusion 파이프라인 미리 로딩
        self.pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            checkpoint_path, torch_dtype=torch.float32,
            safety_checker=None, requires_safety_checker=False
        )

    def generate_image(self):
        img_name = self.lineEdit_image.text()
        prompt = self.lineEdit_prompt.text()

        # Original 이미지 표시
        rgb_path = f'./image2sketch/dataset/images/{img_name}'
        pixmap_rgb = QPixmap(rgb_path)
        self.label_original_img.setPixmap(pixmap_rgb.scaled(300, 300, Qt.KeepAspectRatio))

        # Generated 이미지 초기 텍스트 표시
        self.label_generated_img.setText("Generating... 0%")

        # Worker Thread 실행
        self.thread = GenerateThread(img_name, prompt, self.pipeline, num_steps=50)
        self.thread.progress.connect(lambda p: self.label_generated_img.setText(f"Generating... {p}%"))
        self.thread.finished.connect(self.display_generated_image)
        self.thread.start()

    def display_generated_image(self, generated_image):
        pixmap_generated = pil2pixmap(generated_image)
        self.label_generated_img.setPixmap(pixmap_generated.scaled(300, 300, Qt.KeepAspectRatio))

# ------------------- 앱 실행 -------------------
def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
    
if __name__ == "__main__":
    run_app()

