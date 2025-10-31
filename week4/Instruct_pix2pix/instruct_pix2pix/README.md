# Result
> * Instruction pix2pix 모델을 main.py로 학습.
>   * 학습 Dataset 구조
>     * Triplet{Origianl 이미지: Sketch 이미지, Instruction: Modified Caption, Edited 이미지: RGB 이미지}
> * 학습된 instruct pix2pix 모델을 inference.py로 추론.
>   * Original 이미지(Sketch 이미지)와 Instruction을 입력으로 받아 이미지 생성. 

---

```bash
instruct_pix2pix/
├── dataset/
│   ├── metadata.jsonl
├── utils/
│   ├── make_metadata.py    # 학습 데이터를 json 형식으로 저장하는 스크립트
│   ├── config.py           # 학습 시 필요한 하이퍼파라미터 스크립트
│   ├── train.py            # Instruct pix2pix 학습 스크립트
│   ├── inference.py        # 학습된 Instruct pix2pix 로드 및 추론 스크립트
├── result/                 # 생성된 이미지 예시            
│   ├── 0.jpg
│   ├── 40.jpg
│   ├── 186.jpg
│   ├── 378.jpg              
│   └── 759.jpg
├── main.ipynb                
└── README.md
```

---

<img width="1941" height="1057" alt="image" src="https://github.com/user-attachments/assets/2e09dcd2-73b5-49af-9b3d-af9217ea5e2c" />
