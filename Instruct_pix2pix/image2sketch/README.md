# Sketch 변환
> * RGB 이미지 -> Sketch 이미지
> * Sketch 변환을 위한 여러 모델을 사용.
>    * utils.py의 select_model 함수로 원하는 모델 다운로드. 
>    * 모델 파일이 lua 형식이기에, 이를 사용할 수 있는 가상환경을 구축.
>  * utils.py의 simplify함수로 간단하게 변환 가능.

---

```bash
image2sketch/
├── dataset/
│   ├── utils.py              # sketch 변환을 위한 모델 다운로드 및 추론 스크립트
│   ├── create_sketch.ipynb   # utils의 simplify 함수를 사용하여 sketch 변환 예시 
│   └── images/               # input images (RGB 이미지)
│   │   └── .gitkeep
│   └── gray/                 # Gaussian Blur 이미지
│   │   └── .gitkeep
│   └── sketch/               # output images (Sketch 이미지)
│       └── .gitkeep
├── img/                      # Sketch 변환 과정 sample 이미지
│   ├── image.jpg
│   ├── gray.jpg
│   ├── gaussianblur.jpg
│   └── sketch.jpg
├── README.md
└── requirements.txt
```

---

<img width="1258" height="397" alt="image" src="https://github.com/user-attachments/assets/4a76e7bd-f55a-4063-982f-747f38878d1e" />

* 기존의 Instruct Pix2Pix은 Original Prompt에서 Instruction과 Edit Prompt를 만들고, Prompt들에 대해 Stable Diffusion + Prompt-to-Prompt를 통해 Input Image와 Edited Image를 생성하여 데이터셋을 구성한다.
  * => 이와 같은 방법은 task에 적합한 데이터셋을 구성하는 데 한계가 존재했다.
  * 위의 이유가 sketch 변환을 통해 이를 original 이미지로 사용하는 까닭이다.
 
 ---
<img width="1896" height="616" alt="image" src="https://github.com/user-attachments/assets/2b313fb3-99e5-4755-8b11-6650632ec336" />
