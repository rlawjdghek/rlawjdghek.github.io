---
title:  "[CVPR2021]Distilling Knowledge via Knowledge Review"
excerpt: "[CVPR2021]Distilling Knowledge via Knowledge Review"
categories:
  - Paper & Knowledge
  
tags:
  - Knowledge Distillation
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-21T22:33:00-05:00
---

저자분들에게는 미안하지만 리뷰운이 있었을 것으로 보인다. 이보다 더 성능좋은 논문이 있는데 SOTA라고 언급한 것부터 잘못되었다. 

### Abstract 
여러 KD기법들이 있지만 그 중 feature transformation에 포함되는 논문이다. 하지만 단순히 지식을 전달하는 것이 아닌 사람이 학습하는 메커니즘으로 학습한다고 한다.

### Introduction
기존의 문제점을 저격한다. KD를 예로 들면, 같은 level에 있는 teacher와 student끼리 전이를 진행한다. 또한 FITNET도 다중의 layer에서 학습을 하지만 같은 level끼리 전달하는 것은 변하지 않는다.
하지만 이 논문에서 제시하는 것은 이럴 때 bottleneck이 생겨 distillation을 제대로 하지 못한다는 것. 

결론부터 말하자면, 논문에서 제시하는 방법은 teacher의 low-level feature를 student의 high-level에 전달하는 것이다.
왜냐하면, 사람의 학습 패턴을 볼 때, 우리가 얼핏 초,중고등학교 때 배운 지식은 나중에 되어서야 쉽게 이해가 되는 것을 알 수 있다. 이를 human learning curve라는 용어로 있어보이게 설명을 한다. 
이러한 사실에 기반하여, teacher의 multi-level feature를 student의 하나의 레이어에 정보를 주어야 하는데, 이는 너무 극단적이다. 따라서 메소드 섹션에서는 student레이어 중 유일하게 하나를 사용하는 
것이 아닌 변형된 방법을 사용한다.

이를 통틀어서 저자들은 "review mechanism"이라고 명했다. 즉, student가 상항 이전에 학습한 것을 refresh하는 것으로 부연설명할 수 있다. 사람으로 예를 들면, 각 스테이지는 공부하는 날이고, 
각 날마다 전에 배우넋을 복습한다고 생각할 수 있다. 또한 복습한 것을 토대로 새로운 것을 배울 때 더 잘 학습 할 수 있다.

하지만 문제는 이러한 learning process에서 어떻게 잘 주는지인데, 이를 해결하기 위해 residual learning, ABF, HCL모듈을 제시한다.

정리하자면, review mechanism으로 전체적인 학습에 틀을 잡음 -> 잘 전달하기 위해 ABF, HCL 모듈을 개발함. -> 다양한 데이터셋 실험. 

### Related Work
별거 없음. 이 논문에서는 KD를 one-stage feature distillation (RKD, CRD, FitNet, ...)과 multi-stage feature distillation(AT, FSP, ...)으로 나누었다.  


### Method
introduction에서 설명한 것과 같이 먼저 review mechanism을 설명하고 저자들이 제안한 모듈을 설명한다.
 
##### Review Mechanism
그냥 수식만 복잡하게 쓰고 결론은 아래 그림과 같이 훈련하는것을 목표로 한다. 화살표가 직관적으로 이해 안되게 그려놨지만 앞에 내용읽었으면 어떻게 향하는 지 알수 있다. 범례를 참고하여 화살표가 그냥 feature가 변하는 
forwarding임을 명심하자.
![](/assets/images/2021-08-21-KD_knowledge_review/1.PNG)

##### Residual Learning Framework
Residual Learning을 설명하기 전에, 위의 메인 그림에서 4개의 subfigure가 있는 이유는

(a): Review Mechanism을 위한 큰 틀.

(b): 더 잘 전달하기 위핸 새로운 구조.

(c): cost에서 이득을 보는 구조.

(d): 더 효과적으로 전달하는 구조.

형식으로 발전했다고 보면 된다. 즉, 위의 그림은 연계되는것이 아니라 (a)에서 (d)로 가는 과정을 나타낸 것이다. 
논문에는 수식이 복잡하게 나와있는데 이건 이해할 필요가 없어보인다. 그냥 (d)의 그림을 수식으로 나타낸것. i, j에 대한 설명도 없어서 $\mathcal{M}$이 왜 2변수를 갖는지도 모르겠다. 
Residual Learning을 잘 설명하는 한 예시가 있는데 이것만 읽고 넘어가면 될 것 같다. 

예를 들어, student의 4번째 feature가 student의 3번째 feature와 합쳐지고, 이 합쳐진 것이 teacher의 3번째 feature과 같게 된다고 할 때 이를 수식으로 나타내면,
$student_3 + student_4 = teacher_3$ 이므로 student_4는 $teacher_3 - student_3$라고 할 수 있다. 또한 이는 $student_4$가 $teacher_3$를 배움, 즉, teacher의 low-level이 student의 high-level로 갔음을 알 수 있다.
따라서 residual learning은 teacher의 low-level을 student의 high로 줄 때 더 효과적이라 주장한다. 

##### ABF and HCL
![](/assets/images/2021-08-21-KD_knowledge_review/2.PNG)
그림이 제일 이해하기 쉽다. 
ABF는 
1. resize
2. concatenation
3. multiply
4. add
그림 보면 된다. 중요한 것은 direct로 학습을 하지 않고 adaptive로 했는데 이는 어덯게 잘 전달하는 지도 또한 중간 과정을 더 복잡하게 하여 전달 조차 학습으로 돌리는 효과를 주었다.

HCL은 그냥 그림처럼 pyramid pooling을 사용하였다. 왜냐하면 이것또한 direct로 정보를 전달하면 각 level의 feature가 멀수록(teacher의 1을 student의 4로) 더 차이가 많이 나기 때문에 teacher의 지식을
student가 전체적으로 받지 못하는 현상이 밣생한다. 따라서 Pyramid 구조로 더 유연하게 distillation을 해 준다.


### Experiment
가장 문제가 되는것은 그다지 성능이 높지 않은데 최신 논문을 사용하지 싣지 않아서 성능이 제일 높게 나와보이는 것. 하지만 novelty가 있는것은 object detection과 segmentation에서 실험한 것이다. 결과는 당연히 제일좋음. 
 


 

 




