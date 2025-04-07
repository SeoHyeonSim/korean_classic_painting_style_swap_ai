# 서양화 - 동양화 전환 생성 AI 모델 
AI를 통한 이미지 생성과 화풍 변환 서비스가 많은 주목과 인기를 끌게 되면서, 이미지 생성 및 스타일 전환 모델의 발전이 빠르게 이뤄지고 있습니다. 화풍 변환 모델의 경우, 타깃이 되는 작품의 색감, 붓터치 등의 특징에 대한 모델의 학습도, 그리고 스타일을 변환하는 이미지 내 사물을 화풍에 맞게 변환하는 것이 성능 평가의 지표라고 할 수 있습니다. 이러한 특징을 염두하며 모델에 투입할 데이터셋을 조사하던 도중, 서양화와 한국화의 특징과 두 분야 간 화풍 차이에 주목하게 되었습니다. 색체와 명암이 뚜렷하게 나타나는 서양화와 여백과 먹을 사용해 묘사하는 한국화의 특징을 분석하면서 두 화풍을 서로 교환하여 이미지를 생성하는 모델을 제작하고자 하였습니다. 본 모델은 서양화 이미지를 투입하여 이를 한국화 화풍으로 변환하는 기능을 수행합니다. 


## Overview

모델은 사전 학습된 VGG19 모델을 사용하여 스타일 및 콘텐츠를 추출하는 CycleGAN 아키텍처로 학습을 진행하였습니다. 

- **Domain A:** 서양화 이미지 데이터셋 
- **Domain B:** 한국화 이미지 데이터셋


---

## Features
- 서양화 스타일과 한국 전통화 스타일 간의 이미지 변환
- 두 개의 제너레이터와 두 개의 디스크리미네이터를 사용한 이미지 변환 학습
- 사전 학습된 VGG19 네트워크를 사용하여 스타일 및 콘텐츠 추출
- GAN 손실, 사이클 일관성 손실, 정체성 손실 함수 구현

---

## Model Architecture

### Generators

**ResNet-based architecture**

- 9개의 Residual Block 사용, Instance Normalization 및 ReLU 활성 함수 적용
- 다운샘플링 및 업샘플링 레이어를 통한 이미지 변환

### Discriminators

**PatchGAN architecture**
- 여러 개의 컨볼루션 레이어와 LeakyReLU 활성 함수 사용
- 해당 이미지가 AI에 의해 생성된 것인지 여부 판별 

### VGG19 Style Extractor

- 사전 학습된 VGG19 네트워크를 사용하여 스타일 및 콘텐츠 특징을 추출
- **Gram Matrix**를 사용하여 스타일 피처를 추출

---

## Loss Functions

- **GAN Loss:** Generator가 생성한 이미지 Discriminator를 얼마나 잘 속이는지 평가.
- **Cycle Consistency Loss:** 이미지를 다른 도메인으로 변환 후 다시 되돌렸을 때 원본 이미지와의 일관성 보장. 
- **Identity Loss:** 이미 변환할 필요가 없는 이미지일 때 손실을 최소화하여 정체성을 유지

---

### Hyperparameters

- Learning Rate: 0.0002
- Batch Size: 1
- Number of Epochs: 300

---

## Image Generation

```python
from PIL import Image
from torchvision import transforms
import torch

# Load image
img = Image.open('path/to/your/image.jpg').convert('RGB')

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
img_tensor = transform(img).unsqueeze(0).to(device)

# Generate image
with torch.no_grad():
    output_tensor = G_A2B(img_tensor)

# Post-process and save
output_tensor = (output_tensor.squeeze().cpu() + 1) / 2
output_img = transforms.ToPILImage()(output_tensor)
output_img.save('result.jpg')
```

---

## Results

![image](https://github.com/user-attachments/assets/3b6e4c91-a6bc-4e5e-bb21-de90a7a1c427)

![image](https://github.com/user-attachments/assets/19207a2d-4a26-4aa1-bace-8db35e3425c2)

![image](https://github.com/user-attachments/assets/c51f2191-4d39-4f6b-8a22-c7915217e020)



---


## Acknowledgments

- This implementation is inspired by the original CycleGAN paper: [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).
- Pre-trained VGG19 model from PyTorch.

---

