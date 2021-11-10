---
title:  "[CVPR2021]Data-Free Knowledge Distillation For Image Super-Resolution"
excerpt: "[CVPR2021]Data-Free Knowledge Distillation For Image Super-Resolution"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-11-11T15:04:00-05:00
---

CVPR2021 논문 중 KD 관련 논문을 찾아 보던 중 발견한 논문. 모바일에서 많이 사용하는 기술인 super resolution분야에서 모델의 크기가 커서 크기를 줄이는 방법을 KD로 풀어내었다.

# Abstract & Introduction
여러 API에서 Super resolution을 활용하고 있는데, 이를 이용하여 더 작은 네트워크를 만드려면 어떻게 해야할까? 에 대한 답을 낸 논문이다. 즉, 우리가 사용할 수 있는 모델이 
훈련 데이터로 어떤 것을 썻는지 모르고, teacher model의 output만을 알고 있을 때, 이 teacher network를 사용하여 더 작은 네트워크를 만들어야 한다. 중요한 것은 훈련데이터를
모른다는 점인데, 저자들은 JFT같은 데이터가 공개되어 있지 않지만 여러 API에서 사용되었다는 것을 추측 해 보았을 때, student 모델을 훈련시키기 위해 JFT 데이터가 없어도 JFT데이터로 훈련한 teacher를 
이용하여 student를 훈련해야 할 필요성을 강조한다. 이러한 data-free knowledge distillation은 이전부터 조금씩 연구되어 왔지만 super resolution에 적용하여 필요성을 강조하는 내용이 논문의 값어치를 더욱 
높인 것 같다. \

# Data-Free Learning for Super resolution
먼저 super resolution에서의 KD loss는 아래와 같다. \
$\mathcal{L}\_{KD} = E\_{x\in p_x(x)}\[||\mathcal{T}(x) - \mathcal{S}(x)||_1\]$

이 논문의 가장 큰 기여는 훈련 데이터가 없다는 것이다. super resolution은 훈련을 위해 고해상도 이미지를 필요로 한다. 이 고해상도 이미지를 저해상도로 만들고, 이 저해상도 이미지를
입력으로 하기 때문에 훈련데이턱가 없는 것은 일반적으로 생각할 때 아주 모순적인 상황이다. 저자는 이를 해결하기 위해 teacher network를 이용하여 훈련 데이터를 생성하는 것부터 시작하였다. \
$\mathcal{L}\_{KD} = E\_{x\in p_x(x)}\[||\mathcal{T}(x) - \mathcal{S}(x)||_1\]$
