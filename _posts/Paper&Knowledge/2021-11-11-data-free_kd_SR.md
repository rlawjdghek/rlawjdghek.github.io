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
높인 것 같다.

# Data-Free Learning for Super resolution
먼저 super resolution에서의 KD loss는 아래와 같다.\
$\mathcal{L}\_{KD} = E\_{x\in p_x(x)}\[||\mathcal{T}(x) - \mathcal{S}(x)||_1\]$\
위의 식에서 $p(x)$는 훈련데이터의 분포를 나타낸다. 즉, 우리는 super resolution에서 kd를 적용하기 위해서는 (classification에서도 마찬가지) 훈련 데이터가 필요하다. \
하지만 현재 상황이자, 이 논문의 가장 큰 기여는 훈련 데이터가 없다는 조건이다. super resolution은 훈련을 위해 고해상도 이미지를 필요로 한다. 이 고해상도 이미지를 저해상도로 만들고, 이 저해상도 이미지를
입력으로 하기 때문에 훈련데이턱가 없는 것은 일반적으로 생각할 때 아주 모순적인 상황이다. 저자는 이를 해결하기 위해 teacher network를 이용하여 훈련 데이터를 생성하는 것부터 시작하였다
(기존의 연구들은 classification 분야에서만 이러한 상황을 적용하였다). \
알고리즘부터 한글로 적어보면,
1. random variable z, generator G로부터 $G(z)$를 생성한다.
2. Teacher $\mathcal{T}$를 통과하여 $\mathcal{T}(G(z))$ 고해상도 이미지를 생성한다. 
3. downsample을 적용하여 이미지를 저해상도로로 만든다. $R(\mathcal{T}(G(z)))$
4. 이 저해상도는 훈련데이터와 비슷해야 한다. 따라서 손실함수 $\mathcal{L}\_R = E\_{z \in p_{z}(z)}\[\frac{1}{n}||R(\mathcal{T}(G(z))) - G(z)||_1\]$를 사용한다.
이 알고리즘에서 중요한 가정은 **super resolution 분야에서 고해상도 이미지는 스스로가 gt이고 down sample하면 훈련 데이터가 된다는 것이다. 즉, 어떤 모델이 완벽하게 훈련 데이터를 
학습하였을 때 저해상도의 훈련데이터가 모델을 통과하여 고해상도 이미지(gt)가 생성되고 이를 다시 downsample하면 입력 이미지가 된다.**\
하지만 이 손실함수만 사용하였을 때에는 generator가 mode collapse로부터 벗어나기 힘들어 adversirial loss를 추가하였다. mode collapse가 일어나는 이유는 논문에 서술하지 않았으나,
개인적인 생각으로는 teacher를 통과한 후 downsample되는 연산은 결국 identity function과 같으므로 generator가 계속 같은 이미지만 생성해도 loss를 충분히 줄일 수 있을듯 하다. 
따라서 adversirial loss를 추가하였다. teacher와 student의 output을 같게하는 \
$\mathcal{L}\_{KD}=E\_{x \in p\_x(x)}\[||\mathcal{T}(x) - \mathcal{S}(x)||_1\]$\
를 추가하였다. 