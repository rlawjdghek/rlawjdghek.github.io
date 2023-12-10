---
title:  "[ECCV2020]BroadFace: Looking at Tens of Thousands of People at Once for Face Recognition"
excerpt: "[ECCV2020]BroadFace: Looking at Tens of Thousands of People at Once for Face Recognition"
categories:
  - Paper & Knowledge
  
tags:
  - Representation Learning 
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-10-04T17:04:00-05:00
---

Representation Learning에서도 mining에 해당하는 논문이다. 하지만 이것만으로는 성능이 안나왔는지 베이스 로스를 Arcface로 잡고 추가적인 mining 기법을 붙여 sota를 달성하였다
카카오에서 저술한 논문. 한가지 아쉬운 점은 ablation study를 다양한 loss에 대하여 시도해 봤으면 이 방법이 모든 loss나 제시된 방법에 대하여 일반적으로 잘 나오는지 보여줄 수 있었을 것이다. 

# Abstract & Introduction & Related work
특별한 것은 없다. 다만 이 논문에서 대체적으로 다루는 related work는 모두 Arcface, Cosface 등 loss를 다룬 논문이고, 나중에 Methods 파트에서 나오겠지만
Arcface를 베이스 삼고 훈련 방식만 바꾼 것이라 loss와는 관련이 크게 없어보인다. 하지만 성능 향상이 아주 높아 결과적으로는 ECCV에 실렸다. 

# MEthods
### Typical Learning
지금까지의 거의 모든 딥러닝 프레임워크는 기본적으로 미니배치의 개념을 사용하여 Gradient Descent를 한다. 하지만 가장 처음 미니배치를 배울때 처럼 모든 데이터를 보는 것이 아니기 때문에 편향이 커질 수 있고,
최적화 관점에서 global optimum에 접근하기 어려울 수 있다. 특히나 facial recognition 분야에서처럼 많은 identity = class를 가지고 있는 데이터는 미니배치가 클래스 수에 비해 너무 적기 때문에 더 악화될
수 있다. \
용어부터 정리하자.
1. $x$: an input
2. $X$: mini batch
3. $e$: an embedding vector
4. $W$: an weight matrix
5. $\mathbb{E}$: embedding vector의 queue
6. $\mathbb{W}$: weight matrix의 queue

 
### BroadFace
방법론이 간단하기 때문에 2가지 개념만 알면 된다.
##### 1. Queuing past Embedding vectors
미니배치를 늘리기 위해서 우리가 주목해야 하는 것은 한번에 얼마나 많이 backprop을 해야 하는 것이다. 하지만 모델에는 BatchNorm등으로 한번에 들어오는 미니배치에 대한 통계값으로 결정되는 요소들이 있으면서
원천적으로 이러한 것들은 미니배치를 늘리지 않는 이상 해결 할 수 없다. 따라서 논문에서는 facial recognition의 딥러닝 모델의 기본적인 형태인 1. feature extraction 2. classifier로 나누어 접근하였다.
이렇게 나누면 세부적인 모듈들에 대한 정보를 건너뛸 수 있으므로 저자들의 메인 해결점인 미니배치를 늘린다는 주장에서 파생되는 원초적인 문제들을 해결 할 수 있다. \ 
먼저 feature extraction을 통과한 input은 embedding vector로 변하고 이때의 dimension은 보통 128 ~ 1024를 사용한다. image의 크기가 3x112x112인 것을 감안 하면 거의 70~300배의 차이가 나므로 
좋은 메모리 효율을 자랑한다.
 
우리는 한번의 iteration마다 feature extraction에서 나온 embedding vector를 $\mathbb{E}$에 저장하고, classifier의 weight matrix $W$를 $\mathbb{W}$에 저장한다. weight matrix의 용량이 약간 크므로 
결과적으로 저장할 수 있는 embedding vector의 크기는 약 40배 정도라고 쓰여있다. 

모델을 업데이트 할 때에는 feature extractor는 이전과 같이 미니배치로만 훈련을 하지만, classifier에서 지금까지 모은 feature embedding vector와 weight matrix를 사용한다. 
즉, 여기서 집중하는 것은 feature extractor가 아닌 classifier이다. classifier에서 더 많은 identity를 backprop하면서 편향이 작아지고, 최적의 해에 더 가까워 진다는 논문의 주장이 약한감이 있다.
결국 추론때 사용하는 것은 feature extractor이므로... feature extractor를 한번에 훈련하지 못한 이유는 추측상으로는 feature extractor를 통과한 feature embedding vector가 queue에 쌓이는데 이 오래된 
vector들로 나중에 업데이트르 하면 오히려 손상이 될 수 있다고 생각함. classifier도 손상이 있지만 아래와 같은 방법으로 극복하였다.

##### 2. Compensating past Embedding vectors
feature embedding을 모두 저장 해둔 다음 한방에 classifier를 업데이트 하는 것보다 더 중요한 것이 있다. 위에서 추측한것처럼 feature extractor는 업데이트를 하지 않지만 classifier는 업데이트를 하는 대신 
계속 업데이트 되는 모델에 대하여 보상을 해주어야한다. 간단히 말해서, 기존의 embedding vector ($e_i^-$)에 특정한 보상 ($\rho$)를 더하여 새로운 vector ($e_i^*$)를 만든다. 그 다음 MSE로스로 최적화를 진행.
이 손실함수를 최적화 하면 새롭게 보상받은 $e_i^\*$와 이번 iteration에서 출력된 $e_i$와의 차이를 최소화 한다. 따라서 어느정도 업데이트 방향으로 진행된 과거의 embedding vector를 구할 수 있다.\
수식으로 나타내면 아래와 같다.\
$minimize J(\rho(y)) = E_x\[(e_i^\* - e_i)^2 | y_i = y\] = E_x\[(e_i^- + \rho(y) - e_i)^2 | y_i = y\].$ \ 
일차 미분하면, $\frac{\delta J}{\delta \rho(y)} = E_x\[2 (e_i^- + \rho(y) - e_i) | y_i = y\]$. \ 
따라서 보상 $\rho(y)$ 은 $E_x\[e_i | y_i = y\] - E_x\[e_i^- | y_i = y\] \approx \lambda (W_y - W_y^-)$ \이 된다. \
embedding vector 별로 scale이 다르므로 $\lambda$를 $\frac{\parallel e_i^- \parallel}{\parallel W_{y_i}^- \parallel}$

따라서 최종적으로 손실함수는 \
$\mathcal{L}\_{encoder}(X) = \frac{1}{|X|} \{ \Sigma\_{i \in X}^l(e_i)\}$\
$\mathcal{L}\_{classifier}(X \cup \mathbb{E}) = \frac{1}{|X \cup \mathbb{E}|} {\sum\limits_{i \in X}l(e_i) + \sum\limits_{j \in \mathbb{E}}l(e_j^*)}$\


# Experiments
기본적인 성능은 넘기자. BroadFace가 거의 모든 데이터에서 독보적으로 잘 나왔다. 

![](/assets/images/2021-10-04-BroadFace/1.PNG)
### Size of Queue.
큐에 들어가는 embedding vector의 수를 먼저 보면, 데이터가 많을수록 성능이 향상된다. 하지만 그래프에서도 볼 수 있듯이, compensation이 없으면 확연히 떨어짐.
 





















