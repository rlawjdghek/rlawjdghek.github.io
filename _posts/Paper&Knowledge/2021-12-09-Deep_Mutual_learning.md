---
title:  "[CVPR2020]Online Knowledge Distillation via Collaborative Learning"
excerpt: "[CVPR2020]Online Knowledge Distillation via Collaborative Learning"
categories:
  - Paper & Knowledge
  
tags:
  - KD
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-12-10T15:04:00-05:00
---

# Abstract & Introduction
Teacher-student KD와 self-KD의 중간이라 할 수 있는 online KD의 종류이다. Online distillation은 teacher와 student 모델이 적용되기는 하지만, 전통적인 teacher-student KD의 
최대 단점인 성능이 더 좋은 teacher를 훈련시키는 것을 보완하기 위해 teacher를 student로 사용한다 (sub-student라고 명명). 또한 online이라는 이름에 맞게 모든 네트워크는 초기화 된 상태에서 동시에 훈련된다. 
어차피 cost는 모델이 많아질 수록 배가 될텐데 모든 모델이 스크래치부터 훈련 된다는 것은 약간 비효율적인 것 같다. 미리 훈련된 teacher 모델을 사용하는 것이 성능이 더 높다면 cost면에서는 
비슷하므로 차라리 전통적인 KD를 사용하는 것이 낫다고 생각이 든다. 전통적인 KD와 self-KD와 달리 online KD만이 갖는 특징을 나열하자면 아래와 같다. 
1. 모든 네트워크는 스크래치부터 훈련된다. 
2. teacher와 student의 모델 구조가 같다. (on-the-fly, 줄여서 ONE 논문은 얕은 몇개의 레이어를 공유한다.)
3. 따라서 최종 추론에 쓰이는 모델의 성능이 가장 높다. 
4. cost는 sub-student를 1개만 사용한다는 가정하에: 전통 KD > online KD > self-KD 순으로 정립된다. 

하지만 online-KD의 최종 목표는 self-KD와 비슷한데, 결국 최종 추론에 사용되는 모델의 validation 성능이 가장 높으므로 이 모델의 훈련을 도와주는 헬퍼들이 존재한다. 이 성능이 낮은 헬퍼들이 
메인 모델을 훈련 시키기 위해 어떠한 정보를 주느냐 또는 이런 더 안 좋은 모델이 좋은 모델에 정보를 주는 것이 가능한가가 가장 큰 이슈다. 

TF-KD에서 보여주었지만, 성능이 안 좋은 모델이 좋은 모델에 regularizer 역할을 하기 때문에 hard label만을 사용하는 것보다 도움이 되는것은 사실임에 별개로, 이 논문에서는 이 질문만 던지고 결국에는 sub-student의 앙상블을
 사용하여 성능이 더 좋은 모델을 만든 뒤 최종 모델을 훈련 시켰다. 이렇게 모델을 앙상블 한 것을 collaborative learning이라 함. 논문의 contribution을 정리하자면 아래 3가지와 같다.
1. Collaborative learning 기반의 KD 방법론을 제시하였다. -> 사실 ONE논문이 더 노벨티 있어보이지만, 이 논문또한 다양한 방법 (4가지)으로 변형 될 수 있다는 것을 보여주었다.
2. 전통적인 teacher-student와 달리 one-stage로 soft target을 만들어 훈련하는 방법제시. 
3. Perturbation에 robust함을 보여주었고, 이것이 성능 향상에 도움이 된다. 

# Collaborative Learning for Knowledge Distillation
![](/assets/images/2021-12-10-CLKD/1.JPG)
Soft target을 자동적으로 만드는 KDCL을 제시한다. 전체 프레임워크를 한문장으로 요약하자면, **이미지를 다른 seed로 augmentation해서 각 sub-student를 훈련 시키고 도출된 logit을 앙상블하여 각 sub-student를 다시
KL로 훈련시킨다.** 즉, 여러개의 student 모델을 이용하여 좋은 output을 만든 뒤 이걸로 다시 학습하는 아주 간단한 방법이지만 저자들은 이를 변형하여 soft target을 생성하는 4개의 버전을 만들었다. 
### KDCL-Naive
m개의 sub-student중 가장 cross-entropy가 낮은 output을 골라서 teacher로 사용한다.

### KDCL-Linear
m개의 sub-student의 output에 각각 다른 상수 값을 곱한뒤 더해서 output을 만든다. 이 때 길이가 m짜리 벡터인 $\alpha$와 전체 logit을 concatenate한 $Z$를 곱하서 행렬연산으로 구할 수 있다.

### KDCL-MinLogit
먼저 m개의 sub-student에서 m개의 logit을 뽑는다. 그리고, gt가 c번째 클래스라고 가정 할 때, 각 m개의 logit에서 c클래스에 해당하는 logit값을 뺀다.
그러면 c번째 클래스의 logit값은 0이 되고, 만약 c번째 클래스의 logit값이 가장 컸을때는 나머지가 모두 음수, 아닐때는 어떻게 되는지 모르지만, softmax를 취하면 결국 logit에 상수를 빼는 것이므로 
원래와 값이 같게 된다. 그 다음 각 클래스에 대하여 가장 작은 logit값을 뽑아 soft target으로 만든다. 달라지는 것은 m개의 logit 각각의 c번째 클래스 logit값에 대해서 연산을 진행했으므로 각각의 logit에 대한
softmax값은 같지만, 클래스 별로 가장 작은 logit을 뽑고 softmax를 하면 c번째 클래스 (무조건 0을 가짐)를 제외하고는 더 작아질 수 있다. 하지만 문제는 overfitting 되는 soft target을 만든다는 것..

### KDCL-General
KDCL-Linear의 연장선이라 보면 되는데, 가중치를 학습으로 얻는 것이 아닌, 정의된 에러를 최소화 시키는 함수로 정한다. 이 알고리즘에서는 먼저 generalization error를 아래와 같이 정의한다.
![](/assets/images/2021-12-10-CLKD/2.JPG)
어떤 모델이 예측한 값과 gt와의 차이를 측정하고, 이를 KDCL-Linear에 적용하면 아래와 같이 전체 generalization error를 구할 수 있다. 
![](/assets/images/2021-12-10-CLKD/3.JPG)
C는 아래와 같이 정의 된다. 
![](/assets/images/2021-12-10-CLKD/4.JPG)
라그랑지안 승수법을 활용하여 최적의 w를 구하면 아래와 같고, 나머지는 KDCL-Linear과 같다. 하지만, 이는 매 iteration이 더 정확하다는 한계점을 같고, epoch 단위로 w를 업데이트 하게 되면 cost가 증가한다는 단점이 
존재한다. 
![](/assets/images/2021-12-10-CLKD/5.JPG)

# Experiment
실험에서 주목할 만한 것은 크게 없어서 나열로 정리.
1. KDCL-Linear와 KDCL-Minlogit이 대체적으로 성능이 좋다. 이론상으로는 KDCL-General이 Linear보다 좋아야 하지만, 에폭마다 w를 구하기 떄문에 안좋은듯.
2. sub-student 모델이 증가할수록 성능이 높아진다. 하지만 증가할수록 성능 향상은 미미해진다. 
3. KDCL로 같은 구조가 아닌 서로 다른 구조에 대해서 진행 할 때, overfitting을 완화하는 효과를 준다. 논문에서는 WRN에서 ResNet18로 주었는데, WRN의 훈련 acc가 ResNet보다 높은 반면, valid에서는 더 낮았음.
4. CIFAR100에서는 KDCL-Naive가 성능이 가장 높은데 이유는 앙상블 기반 방법은 쉬운 데이터셋에서 오히려 overfitting을 도와주는 효과를 줄 수 있다고 추측. 
