---
title:  "[AAAI19]Data-Distortion Guided Self-Distillation for Deep Neural Networks"
excerpt: "[AAAI19]Data-Distortion Guided Self-Distillation for Deep Neural Networks"
categories:
  - Paper & Knowledge
  
tags:
  - KD
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-12-13T15:04:00-05:00
---

이 논문은 introduction과 related work에 KD를 잘 정리해 두어서 논문 작성할때 참고하면 좋을 것 같다. 

Self-KD 논문들은 각각 student와 데이터, hard label만 있는 상황에서 크로스 엔트로피 이외에 어떤 항을 추가로 할지를 결정한다. PSKD는 시간을 축으로 삼아 이전 에폭에
훈련된 모델을 teacher로 하였고, TFKD는 크로스 엔트로피로만 훈련된 모델을 teacher, CSKD는 데이터를 반으로 나누어 한쪽을 teacher, 나머지 한 쪽을 student라고 하였다. 
이 논문은 데이터로 접근하였고, 원본 데이터가 아닌 데이터 augmentation이 적용된 데이터를 teacher로 사용하였다.

# Abstract & Introduction
딥러닝 모델의 potential capacity는 consistent global feature distribution과 posterior distribution을 학습함으로서 결정된다고 한다. 다시 말하면, consistent global feature distribution은
feature extraction이 다르게 augmentation되거나 같은 클래스내의 다른 이미지로부터 뽑는 feature의 분포의 일관성을 의미한다. 이는 robust한 feature extractor 일수록, 본질적으로 같은 데이터는 
변형 되더라도 대표적인 representation을 뽑는다는 말. 또한 posterior districution은 앞에서 feature extractor가 뽑은 벡터를 주어진 클래스들의 사후확률 분포로 나타낸다. 결국 이는 자주 사용하는 정확도와 밀접하게
관련되어 있다. 

이렇게 결정되는 capacity는 작은 네트워크에서 작다. 작은 네트워크는 스크래치부터 학습을 진행할 때, 큰 모델보다 학습이 어렵다고 할 수 있다. 학습이 어렵다는 것은 local minima에 빠질 확률이 높다는 말과 동치이고, 우리는 
이 local minima를 벗어나 global minima에 더 원활하게 다가가기 위해서 KD 방법론을 적용하는 것이다. Capacity가 큰 teacher 모델은 hard label보다 학습이 쉽고, 더 robust한 정보, teacher의 예측 값을 주기 때문에
student가 KL Divergence를 통해 더 쉽게 teacher의 예측값을 따라한다고 해석할 수 있다. 또한 종종 student가 teacher의 성능을 넘는 경우도 있는데, 이는 student의 잠재적인 capacity는 충분하지만 기존의 학습이 어려운 
크로스 엔트로피는 local minima로 유도한 것이었고, teacher를 통해 더 좋은 학습 방법을 찾은뒤 크로스 엔트로피가 추가되어 teacher의 성능을 넘었다고 할 수 있다. 또한, teacher를 student 모델로 사용하는
peer-teaching == online-KD는 결국 teacher의 성능을 크로스 엔트로피만 사용한 student의 성능을 넘는 모델을 만들어서 최종 예측에 사용되는 student의 성능을 향상시킨다.
결국 위의 방법론, teacher-student KD, student-student KD의 가정은 teacher의 성능이 student보다 크다는것. 또한 최종 예측에 사용되는 모델의 성능이 teacher보다 낮다는 것이다. 그렇게 때문에 student가 teacher를 따라하는 해석이 성립된다.

하지만 위의 두 방법은 성능개선이 쉽고 해석이 명확하다는 장점이 있지만, 명백하게 단점도 존재한다.
1. 전체 훈련 프로세스가 굉장히 비싸다. 만약 teacher의 모델이 student보다 10배가 크다면 모델을 훈련하는데에 10배의 cost가 더 든다.
2. 만약 어떤 task에서 teacher 모델이 overfitting된다면, teacher를 사용할 수 없다.
3. teacher 모델이 어떻게 student의 성능을 부스팅 하는지 명확하지 않다. student가 단순히 teacher의 성능을 따라한다고 하는 것은 맞지만, teacher가 어떤 이미지에 대해서 틀린 예측을 했다고 할 때, 이것이 항상 옳다고 할 수 없다.

# Method
전체적인 훈련과정을 정리하면 아래와 같다.
1. data augmentation을 통하여 훈련 데이터를 2개로 나눈다.
2. 하나의 global feature extraction으로 representation vector를 얻는다.
3. MMD metric으로 두 데이터의 차이를 줄인다.
4. FC layer와 softmax를 거쳐 두 확률 분포를 얻는다.
5. KL Divergence를 양쪽 모두 해준다.

### The MMD metric for global feature distributions
![](/assets/images/2021-12-13-Data_distribution_self-KD/2.JPG)
논문에서는 기존에는 잘 사용하지 않는 MMD metirc을 소개하였다. 앞에서 소개한 model capacity를 결정하는 feature distribution에 해당하는 것인데, 입력 데이터가 약간 다르므로, 이를 일정하게 맞춰주는 constraint이다.
공식은 아래와 같다. 
![](/assets/images/2021-12-13-Data_distribution_self-KD/1.JPG)

다음으로 KLDiv 로스.
![](/assets/images/2021-12-13-Data_distribution_self-KD/3.JPG)
마지막으로 크로스 엔트로피
![](/assets/images/2021-12-13-Data_distribution_self-KD/4.JPG)

위의 손실함수를 모두 두 쌍에 대해서 계산하면 아래와 같다.
![](/assets/images/2021-12-13-Data_distribution_self-KD/5.JPG)
하이퍼 파라미터 $\lambda=1, \mu=0.0001~0.0005$을 사용하였다. 

# Experiment
실험은 19년도 논문에 비해 빈약한 편이다. CIFAR-10, CIFAR-100, ImageNet에 대한것이 끝. 또한 비교가 다른 모델과 분명하게 이루어진 것도 아니고 단순 베이스라인 (크로스 엔트로피만 사용)과 비교하였다. 
$\mu$는 consistent global feature를 강조하는 파라미터이지만 저자들은 이를 단순 regularizer로써만 바라본다. logit이 같아지는 것은 굉장한 제약이기 때문에 가중치도 매우 작다는 것을 알 수 있다.

# Ablation studies
![](/assets/images/2021-12-13-Data_distribution_self-KD/6.JPG)
이 논문에서 주목할 만한 손실함수는 MMD와 KL이므로 이 두개를 중간에 넣어 비교하였다. 또한 위의 표에서 이 두 손실함수가 없는 two-branch에서는 오히려 resnet32의 성능이 낮아진 것을 비추어 볼 때, 단순 feature extractor를 공유하는 것은
성능향상에 도움이 되지 않는 다는 것을 알 수 있다. 



 