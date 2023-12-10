---
title:  "[ICCV2019] Onthe Efficacy of Knowledge Distillation"
excerpt: "[ICCV2019] Onthe Efficacy of Knowledge Distillation"
categories:
  - Paper & Knowledge
  
tags:
  - Knowledge Distillation
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-14T15:33:00-05:00
---



KD study 3주차에 다룬 논문. 굉장히 이해하기 쉽고 이런 간단한 발견 및 적용 방법이 탑 티어에 실릴 수 있다는 증거로 기억에 남았다. 


### Abstract & Introduction

KD가 출현한 이래로 많은 곳에서 쓰이고 있는데 저자는 이를 좋지 못하게 보았다. 이유는 작은 데이터에서만(MNIST, CIFAR-10, Imagenet의 일부) 도출된 결과를 다루고, 자기들이 돌린 실험에서 같은 결과가 나오지 않았음.

초반부에서는 지금까지 나온 수많은 KD 논문의 문제점들을 제시하고 자신들이 발견한 문제점과 이를 타파할 수 있는 해결책을 제시할 것이라고 명시한다. 

의문점 1. KD를 썼음에도 결과가 향상되지 못한 이유는 무엇인가?

의문점 2. 더 좋은 결과를 도출할 수 있는 teacher-student 조합이 있는가?

의문점 3. 성능을 향상 시킬수 있는 방법이 있을까?


### Results

#### Bigger models are not better teachers & Analyzing student and teacher capacity

이 논문은 딱히 어려운 설명이 없다. 전부다 실험으로만 보여줘서 중간 배경지식은 간단히 넘어가자. 

Bigger models are not better teachers & Analyzing student and teacher capacity

이 논문의 핵심적인 주장 및 방법론이라고 할 수 있다. 게속해서 보여주는 것은 teacher의 크기가 student에 비해 너무 크거나, 훈련, 추론에 쓰이는 데이터가 student에 비해 너무 크면, 성능이 저하되는 것을 볼 수 있다. 저자는 이 문제에 대하여 2가지 가설을 세웠다. 

1. student가 teacher를 따라하지만 KD loss와 정확도가 상관관계가 없어 loss가 줄어도 정확도가 줄지 않는것.

2. student가 teacher를 따라하지 못하는 것.

결론적으로는 2번가설이 맞다. 이것을 증명하기 위해서 KD loss와 accuracy 그래프를 그려보았는데 서로 크기 차이가 다른 teacher-student 모델에서 다른 loss의 차이를 발견 할 수 있었다. 즉, 애초에 loss 차이가 있으므로 그에 따라 accuracy가 
떨어지는 것이다. 이로써 위의 의문점 1,2 에 대한 답이 되었다.

#### Distillation adversly affects training 

의문점 3에 대한 대답이 나오는 파트이다. 먼저 KD를 써서 성능이 저하되는 예부터 다루고 있다. 또한 여기서 Introduction에서 언급하였듯이 대부분의 논문이 간단한 데이터셋을 다루는 것을 지적했다. 
왜냐하면 student의 크기가 Imagenet과 같이 매우 큰 데이터셋을 수용할 정도로 크지 않다면 좋은 결과를 얻기 힘들다는 것이다. 따라서 여기서는 이러한 문제점들과 함께 teacher와 student의 격차가
 많이 날 경우 성능 개선의 해결책을 제시한다. 

저자가 또한 생각한 가설이 student의 capacity가 부족해서 KD loss와 training loss 둘 중 하나가 먼저 끝(수렴) 한다는 것이다. 즉, scratch 부터 학습하는 동일한 student보다 loss를
 minimize하지 못한다는 것이다. 그래서 고안한 것이 teacher의 training을 early stop하는 것이다. 대부분의 결과에서 early stop을 쓴것이 더 좋은 결과를 초래했다. 하지만 early stop을 
 사용한다 하더라도 teacher와 student 사이의 격차가 더 클수록 성능이 저하되기는 마찬가지였다.

#### The efficacy of repeated knowledge distillation

여기서 sequential KD의 개념이 나온다. Sequential KD란 teacher와 student 사이의 갭을 줄이기 위해 중간 사이즈의 모델을 추가하는 것이다. 하지만 여기서의 
결론은 마지막 가장 작은 student와 scratch로 훈련한 student를 비교해도 역시 scratch가 좋다는 것이다. 또한 이전에 논문들은 이 sequential 모델들을 ensemble해서
 성능을 높였다고 하는데 이것또한 그냥 모든 모델을 scratch하는 것보다 성능이 안좋다. 즉 모든 모델이 지식을 계속해서 전달하기 때문에 <u>독립성이 떨어져서 앙상블 효과가 떨어진다.</u> 

이 파트에서 또한 저자가 제시한 early stop을 뒷받침 해준다. 이유는 early stopping 하는 것이 teacher가 지식을 애초에 조금 전달하는 것이므로 더 작은 모델을 
사용한다는 것과 동일한 효과를 낼 수 있다는 것이다. 