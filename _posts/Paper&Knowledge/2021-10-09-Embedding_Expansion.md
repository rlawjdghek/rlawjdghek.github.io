---
title:  "[CVPR2020]Embedding Expansion: Augmentation in Embedding Space for Deep Metric Learning"
excerpt: "[CVPR2020]Embedding Expansion: Augmentation in Embedding Space for Deep Metric Learning"
categories:
  - Paper & Knowledge
  
tags:
  - Representation Learning 
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-10-09T15:04:00-05:00
---

CVPR 2022를 준비하면서 읽은 논문. 방법론도 좋지만 간단한 방법을 논문화 하기위해 실험이 좋은것 같다. 직관적이고 충분히 추론할 수 있는 확장성 있는 실험으로 저자들이 제안한 방법론을 더 돋보이게 해준다. 

# Abstract & Introduction 
Metric learning의 확장된 분야로 loss를 augmentation과 generalization해주는 방법을 제시한다. 개인적인 생각으로는 metric learning에서 완전히 새로운 loss를 제시하는 것은 상당히 어렵기 때문에
기존에 있는 loss에 추가적인 방법을 곁들여 성능을 향상시키는 방법을 채택했다고 본다. 하지만 논문에서 한가지 아쉬운 점은 SOTA를 달성했다고 말하지만 그 기준이 애매하다는 것이다. 
baseline들은 저자들이 제시한 add-on 방법론이 아닌 비교적 옛날에 소개된 방법들이기 때문에 성능차이가 크게 두드러지지 않는다. 하지만 metric learning에서 2개 이상의 data point를 사용하는 loss
에 대하여 적용가능하기 때문에 (논문에서 사용한 base loss는 4개) 실험이 많았기 때문에 표가 부족해 보이지는 않는다.\
Metric learning에서의 성능 증가는 크게 loss와 Mining (=Sampling)으로 나눌 수 있다. Introduction에서는 저자들이 소개한 방법이 Mining 기반이기 때문에 loss에 대한 설명은 굉장히 짧다. 
Mining 전략은 hard sample을 고르는 것이 목적이다. 왜냐하면 쉬운 image pair들은 기존의 유명한 loss들, arcface, triplet 등으로 쉽게 분류가 가능하기 때문이다. 
**하지만 원초적인 문제는 이렇게 hear sample에만 집중하여 sampling 하다보면, 소수의 hard sample에 모델이 편향이 될 수 있다는 것이다.** 따라서 기존 embedding augmentaion은 
GAN이나 Autoencoder를 활용하였으나 이는 시간과 메모리를 많이 잡아먹기 때문에 효율성이 떨어진다. 따라서 독립적인 모델을 제시하기 보다 간단하고 모델과 함께 
end-to-end로 학습 가능한 방법이 필요하다. 

# Methods
저자들이 제시한 방법론은 크게 2가지라 할 수 있다.
1. 2개의 클래스에서 각 클래스에 속하는 2개의 data point로부터 interpolation을 구한뒤, normalization하여 hypersphere에 projection
2. projection된 서로 다른 클래스에서 가장 짧은 distance를 가지는 점, hard negative point 선택. 

Triplet loss, lifted loss, N-Pair loss, MS loss 총 4개의 loss에 대하여 적용한다. 식은 기존의 식과 크게 다르지 않다.

방법론이 간단하므로 아래 그림을 보면 이해할 수 있다. 
![](/assets/images/2021-10-09-Embedding_Expansion/1.PNG)
![](/assets/images/2021-10-09-Embedding_Expansion/2.PNG)
  

# Experiment
### 1. 4개의 로스에 대하여 각각 EE를 붙인 것과 안붙인 것의 성능을 보여줌. NMI와 F1, Recall을 사용함. \ 

* NMI를 예제를 통해서 계산해보자. \
$Y = class labels$ \ 
$C = cluster labels$ \
$H() = entropy$ \
$I(Y; C) = Mutual Information$ \ 
log들은 2를 밑으로 가진다. \
$NMI(Y, C) = \frac{2 \times I(Y;C)}{\[H(Y) + H(C)\]}$ \
![](/assets/images/2021-10-09-Embedding_Expansion/8.PNG)의 NMI를 구해보자.
위의 상황에서는 4가지만 구하면 된다.

1. $H(Y|C=1) = -P(C=1) \sum P(Y=y | C=1) log(P(Y=y | C=1))$
2. $H(Y|C=2)$
























