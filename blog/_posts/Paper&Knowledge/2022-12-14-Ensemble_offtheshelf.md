---
title: "[CVPR2022]Ensembling Off-the-shelf Models for GAN Training"
excerpt: "[CVPR2022]Ensembling Off-the-shelf Models for GAN Training"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-12-14T15:04:00-05:00
---

이것도 마찬가지로 Jun-Yan Zhu가 교신저자로 들어가 있는 논문. 알고리즘 자체는 쉬운데, 굉장히 실용적이다. 최근 CLIP과 같은 pre-trained 모델을 활용하는 논문들이 많이 보이는데, 이를 GAN에도 접목하여 GAN의 성능을 향상시킨 것이다. Diffusion이 나왔다고는 하지만, 아직 해결할 수 없는 고유의 문제점이 있으므로, VQ-VAE처럼 GAN도 계속 발전하다 보면 단점이 없어질 수도 있다. 본 논문도 DiffAug 논문과 비슷하게 GAN 학습을 더욱 안정시키고 성능을 향상시키는 방법이다. 본 논문에서 제시한 방법론을 요약하면 아래와 같다. 

1. 다양한 task에서 다른 목적 (손실 함수)으로 훈련된 모델을 준비한다. 이 모델들은 self-supervised가 될 수도 있고, supervised가 될 수도 있다. 
2. feature extractor는 고정시키고, classifier head를 붙여 이진 분류를 하도록 변형한다. 
3. 기존의 스크래치부터 훈련되는 G와 D는 그대로 훈련한다. 여기에 추가되는 것은 "vision-aided adversarial training"이라 명명된 추가적인 손실 함수이다.
4. 모델 구성은 k-progressive로 한다. k-progressive를 설명하면 아래와 같다.
4-1. 처음에는 pre-trained 모델 집합을 공집합으로 둔다. 단, 모델 후보 집합에는 많은 모델이 있다.
4-2. 미리 정해둔 iteration마다 새로운 모델을 추가하는데, 새로운 모델은 현재 훈련에 사용하는 집합에 없는 모델 중 후보 집합에서 가장 높은 로스를 보이는 D를 갖고 온다.
4-3. 미리 정해둔 k개의 모델이 다 차면 더 이상 후보집합에서 모델을 추가하지 않는다. 

방법론을 이해하는데에 큰 어려움이 없어 코드 구현은 하지 않았다. 

# Abstract & Introduction & Method
최근 컴퓨터 비전 분야에서도 large-scale training이 자연어의 BERT와 같이 강력한 모델을 학습하는데에 사용되고 있다. 지금까지는, classification, object detection에서 pre-trained모델이 일반화된 성능을 보였다. 본 논문에서는 이러한 강력한 feature extractor가 GAN에도 도움이 되는지를 보여주고, 간단한 방법론을 제시한다. 

DiffAugmentation 논문에서도 언급했듯이, GAN의 훈련과정에서 D가 너무 강력하여 결국 훈련 데이터를 외워버리는 오버피팅이 발생한다. 만약 이러한 문제가 classification에서 발생했다면, transfer learning을 고려해 볼 수 있을 것이다. 마찬가지로, GAN에서도 D를 pre-trained 모델로 교체한다면, 결국 feature extractor는 고정되어있으므로 오버피팅 현상이 완화될 것이다. 이러한 직관을 갖고 위에 요약한 방법론으로 구현한다. 크게 방법론이 어렵지 않으므로 자세한 알고리즘은 아래 그림을 참고하자. 

![](/assets/images/2022-12-14-Ensemble_offtheshelf/1.PNG)

알아두어야 할 것은, 알고리즘에서 k개의 pre-trained모델을 사용했다면, 손실 함수는 아래와 같이 k개의 추가적인 D에 대한 손실 함수를 더해주는 것이다. 또한 논문에서 주목한 점은, 기존의 스크래치부터 학습되는 D가 없으면 pre-trained된 모델만으로는 발산한다는 것이다. 이는 pre-trained 된 네트워크들이 스크래치부터 훈련된 G보다 훨씬 강력하기 때문에 학습이 안되는 듯 하다.