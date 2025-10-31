# <p align = "center"> Instruct Pix2Pix </p>  
<p align = "center"><img src="https://github.com/user-attachments/assets/0b203565-369f-40e5-9487-57dc2e253b42" width="1000" height="235"></p>

> 1. RGB 이미지를 스케치 형태의 이미지로 변환  
> (1) OpenCV를 사용해 이미지를 RGB -> Gray로 변환  
> (2) 저장된 Gray 이미지를 Gaussian Blur를 통해 Sketch 이미지로 변환 후 저장  
> (3) Gausssian Blur로 만들어진 Sketch 이미지를 GAN을 통해 더욱 깔끔한 스케치 형태로 변환 후 저장
>
> 2. RGB 이미지에서 caption을 생성  
> (1) CLIP + BLIP으로 이미지에 대한 caption을 생성  
> (2) ChatGPT를 사용해 생성된 caption을 정제
>
> * Dataset{1번: Original Image, 2번: Edit Prompt, RGB 이미지: Edited Image}

---

<img width="1472" height="478" alt="image" src="https://github.com/user-attachments/assets/0689239d-62bd-485b-94e6-240fd9c169ab" />
