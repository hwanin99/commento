# Image Captioning
> * CLIP + BLIP (clip_interrogator 라이브러리)를 사용해서 RGB이미지에서 captioning을 한다.
>   * 해당 caption이 instruction이 된다.
>
> * 위처럼 생성한 caption은 정제되어있지 않다.
>   * 이로 학습 시, 모델 성능 저하될 가능성이 있다.
>   * 따라서, ChatGPT로 caption을 정제된 형태로 다듬는다.  

---

```bash
image_caption/
├── dataset/
│   ├── desc.csv                          # clip_interrogator(CLIP + BLIP)으로 생성한 caption 파일
│   └── modified_desc.csv                 # ChatGPT로 수정한 caption 파일
├── utils/                     
│   ├── captioning.py                     # 이미지에 대한 caption 생성을 위한 스크립트
│   └── edit_caption.py                   # 생성된 caption을 정제하기 위한 스크립트
├── Captioning.ipynb                      # captioning.py 사용 예시
├── Modified_Caption_with_ChatGPT.ipynb   # edit_caption.py 사용 예시
└── README.md
```

---

<img width="1870" height="780" alt="image" src="https://github.com/user-attachments/assets/f46b36fc-d4ce-4ac0-a56c-1638b05bfe6b" />

> * 학습 Dataset 구조
>   * Triplet{Origianl 이미지: Sketch 이미지, Instruction: Modified Caption, Edited 이미지: RGB 이미지}
