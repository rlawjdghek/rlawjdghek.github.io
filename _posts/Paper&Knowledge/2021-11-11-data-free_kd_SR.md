---
title:  "[CVPR2021]Data-Free Knowledge Distillation For Image Super-Resolution"
excerpt: "[CVPR2021]Data-Free Knowledge Distillation For Image Super-Resolution"
categories:
  - Paper & Knowledge
  
tags:
  - KD, Super Resolution
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-11-11T15:04:00-05:00
---

CVPR2021 논문 중 KD 관련 논문을 찾아 보던 중 발견한 논문. 모바일에서 많이 사용하는 기술인 super resolution분야에서 모델의 크기가 커서 크기를 줄이는 방법을 KD로 풀어내었다.
어려운 개념은 안 쓰였는데, CVPR에서 많이 볼 수 있듯이 새로운 task를 기존의 방법들을 참신하게 섞어서 잘 풀어내었다. super resolution에서 훈련 데이터가 없이 더 작은 network를 만들었다. 


# Abstract & Introduction
여러 API에서 Super resolution을 활용하고 있는데, 이를 이용하여 더 작은 네트워크를 만드려면 어떻게 해야할까? 에 대한 답을 낸 논문이다. 즉, 우리가 사용할 수 있는 모델이 
훈련 데이터로 어떤 것을 썻는지 모르고, teacher model의 output만을 알고 있을 때, 이 teacher network를 사용하여 더 작은 네트워크를 만들어야 한다. 중요한 것은 훈련데이터를
모른다는 점인데, 저자들은 JFT같은 데이터가 공개되어 있지 않지만 여러 API에서 사용되었다는 것을 추측 해 보았을 때, student 모델을 훈련시키기 위해 JFT 데이터가 없어도 JFT데이터로 훈련한 teacher를 
이용하여 student를 훈련해야 할 필요성을 강조한다. 이러한 data-free knowledge distillation은 이전부터 조금씩 연구되어 왔지만 super resolution에 적용하여 필요성을 강조하는 내용이 논문의 값어치를 더욱 
높인 것 같다.

### Training data Generator
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
4. 이 저해상도는 훈련데이터와 비슷해야 한다. 따라서 손실함수 \
$\mathcal{L}\_R = E\_{z \in p_{z}(z)}\[\frac{1}{n}||R(\mathcal{T}(G(z))) - G(z)||_1\]$\
를 사용한다.
이 알고리즘에서 중요한 가정은 **super resolution 분야에서 고해상도 이미지는 스스로가 gt이고 down sample하면 훈련 데이터가 된다는 것이다. 즉, 어떤 모델이 완벽하게 훈련 데이터를 
학습하였을 때 저해상도의 훈련데이터가 모델을 통과하여 고해상도 이미지(gt)가 생성되고 이를 다시 downsample하면 입력 이미지가 된다.**\
하지만 이 손실함수만 사용하였을 때에는 generator가 mode collapse로부터 벗어나기 힘들어 adversirial loss를 추가하였다. mode collapse가 일어나는 이유는 논문에 서술하지 않았으나,
개인적인 생각으로는 teacher를 통과한 후 downsample되는 연산은 결국 identity function과 같으므로 generator가 계속 같은 이미지만 생성해도 loss를 충분히 줄일 수 있을듯 하다. 
따라서 adversirial loss를 추가하였다. teacher와 student의 output을 같게하는 $\mathcal{L}\_{GEN} = -log(\mathcal{L}\_{KD} + 1)$를 추가하였다. \
최종적으로 훈련데이터를 만드는 generator의 손실함수는 아래와 같다. \
$\mathcal{L}\_{G}=\mathcal{L}\_{GEN} + w_R \mathcal{L}\_{R}$

### Progressive Distillation
이제 생성된 훈련 데이터로 distillation을 진행한다. 저자들은 이 부실한 훈련 데이터로 student를 teacher로부터 direct하게 훈련되기 어렵다고 한다. 따라서 훈련이 쉬운 더 작은 student를 만들고 
이 모델부터 훈련을 시작한다. 아무래도 생성된 데이터의 다양성이 부족하기 때문에 작은 모델부터 시작하는 듯 하다. 손실함수는 비교적 간단한데, 생성된 이미지를 teacher와 student에 대하여 KD loss를 
사용한다.\
$\mathcal{L}\_{KD\_{s_i}}=E_{z \in p_z(z)}\[\frac{1}{n}||\mathcal{T}(G(z)) - \mathcal{S}_i(G(z))||_1\]$
student를 점점 증가시키는데 어차피 원래의 student network에서 block을 하나씩 더해가는 것이다. 이제 여기까지 보고 그림을 보면 단번에 이해가 된다. 
![](/assets/images/2021-11-11-data_free_kd_sr/1.JPG)

### Optimization
이제 훈련과정을 보자. 알고리즘이 정확하게 나와있어서 이해하기 쉽다. 
![](/assets/images/2021-11-11-data_free_kd_sr/2.JPG)
간단히 정리하면,
1. G를 고정, kd loss로 i번째 student 업데이트
2. i번째 student고정, $\mathcal{L}_G$로 G를 업데이트
3. 어느정도 iteration 후에 i증가, 대신 i번째 student가 i+1번째 student에 해당되는 weight 그대로 복사

# Experiment
![](/assets/images/2021-11-11-data_free_kd_sr/3.JPG)
기존에 이러한 문제를 푼 논문이 없어서 비교대상이 없다. 논문에서는
1. teacher 
2. 훈련데이터가 있는 student
3. generator를 사용하지 않고 일반 uniform distribution에서 뽑은 noise 이미지 사용
4. Ours
5. Bicubic
를 비교했다. Bicubic은 무엇을 사욜한건지 명시하지 않았다. \
신기하게도 noise를 학습해도 훈련이 된다는 것. 특정 이미지가 아닌 노이즈 이미지도 super resolution에 아주 미미하게는 학습이 되는듯 하다.  