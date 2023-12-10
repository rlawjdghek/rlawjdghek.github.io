---
title:  "[ICLR2021]Knowledge Distillation Via Softmax Regression Representation Learning"
excerpt: "[ICLR2021]Knowledge Distillation Via Softmax Regression Representation Learning"
categories:
  - Paper & Knowledge
  
tags:
  - Knowledge Distillation
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-18T15:33:00-05:00
---

AAAI 준비하면서 SR성능이 발목을 잡아 한번 더 읽는 겸 중요한 논문이라 생각되어 기록한다. 

### Abstract & Introduction
FitNet이나 Attention transfer와는 달리 마지막 레이어에 집중한다. 이것이 이 논문의 가장 큰 핵심. 


### Introduction
여러 KD를 언급하면서 그냥 단순히 설명만 한다. 다른 KD논문과 같이 별다르게 주제가 중요하다는 특별한 의미는 없음. Supervision과 Unsupervision에서 널리 사용되고 있는 테크닉이다. 라는 정도만 제시. 
Main Contribution 4가지 제시.
1. student의 penultimate layer에 집중한 손실 함수 제시.
2. 논문의 문장을 그대로 쓰자면, "we propose to decouple representation learning and classification and utilize the teacher's pre-trained classifier, which is achieved with a simple $L_2$ loss."
그림에서 볼 수 있듯이 새롭게 도입한 레이어인데 이 레이어를 통과한 teacher와 student의 오차를 줄이는 손실함수를 개발. 논문 중간중간에서는 student는 teacher를 따라가야 한다고 주장하는데 왜 decouple이라는 말을 사용했는지는 잘 모르겠다.


### Related Work
크게 두가지로 나누었다. 1. teacher의 feature를 직접적으로 전달하는것. 2. teacher와 student의 관계를 매칭.
 
예를 들어, 1번 경우는 원초적인 Hinton의 KD논문에서 제시한 KD loss와 FitNet, AT에서 제시한 방법론 처럼 직접적으로 tensor끼리의 오차를 구하는 것이고,
 
2번경우는 Feature relationship transfer라고 해서 RKD처럼 지식을 전달하기 위해 tensor를 직접적으로 계산하는 것보다 tensor의 관계를 포착. 

3번경우도 짧게 설명하는데 이 경우는 CRD로 적었다. CRD는 가장 최근에 나온 SOTA모델이므로 언급하였지만 CRD와 직접적으로 비교하기 위해 자신들이 제시하는 방법론과 크게 연관있지 않다. = novelty가 있다.고 설명한다. 


### Method
결론적으로 가장 중요한것은 abstract에서도 언급하였듯이 student입장에서 중간중간 정보를 받으면 성능이 오히려 떨어진다는것. self-distillation에서는 오히려 좋아지는데 이건 teacher를 사용하지 않아서 논외로 하는 것 같다. 
총 2가지 loss $L_{FM}$, $L_{SR}$을 제시한다. 

FM은 FITNET과 같이 중간 tensor를 비교하는 것인데, 논문에서 언급하듯이 **하나의 네트워크는 각각의 block들이 순차적으로 제 기능을 하면서 
서로서로 도움이 되는데 개별 block들이 외부에서 들어오는 정보로 학습된다면 independence가 커져서 오히려 방해가 된다.**라는 단점을 지적했다. 따라서 논문에서도 이 점을 검증하기 위해 실험적으로 conv2, conv3, conv4에 각각 
$L_{FM}$을 연결하여 결과가 conv2 < conv3 < conv4임을 보여준다. 

$L_{SR}$은 그냥 마지막에 두 feature정보를 합치는 건데 좀 더 스무스하게 하기 위해 레이어 하나 추가하고 teacher와 student의 출력이 이 레이어를 지나가 새로운 출력을 만들어 내고 그 두개의 출력의 오차를 최소화. 
![](/assets/images/2021-08-18-softmax_regression_KD/1.PNG)

### Ablation Studies 
자신들의 방법론에서 사람들이 제시할만한 추가 실험들을 보여줌.
1. 제시한 2가지 손실함수들이 모두 유효한가?? -> 당연하게도 추가 실험으로 유효한지 보여줌. 
2. loss들이 의미하는 바는 무엇인가?? -> SR은 무조건 사용하고 FM도 마지막 레이어에 사용해라. 즉, 자신들이 제시한 두 손실함수는 어디에나 달 수 있는데 본 최종 모델에서는 둘다 마지막에 달았음. 
3. Teacher-student의 similarity가 얼마나 비슷해지는지?? -> 위에서 언급한 decouple과 관련된 의문점이다. 여기서는 teacher와 student를 최대한 비슷하게 하는것이 KD라 하였는데 분리라는 말이 이해가 안간다. 

결론적으로 다양한 실험으로 모두 말이 된다는 것을 보여주었다. 

### Comparison With State-of-the-art
다양한 벤치마크 데이터셋에 대하여 성능 비교. 사용한 데이터셋은 CIFAR-10, CIFAR-100, Imagenet, Facial landmark.
중간에 Binary neural network가 나오는데 이거는 모델의 weight와 activation 된 값들이 2개, 즉, 1과 -1로 되어있음. 추론 속도가 증가하고 전력 손실이 줄어드는 효과가 있다. 

### Conclusion 
방법론 자체가 간단하고 주장한 바를 그대로 적었다. 

  

