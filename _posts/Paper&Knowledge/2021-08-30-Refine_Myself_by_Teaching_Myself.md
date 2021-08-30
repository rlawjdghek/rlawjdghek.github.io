---
title:  "[CVPR2021]Refine Myself by Teaching Myself: Feature Refinement"
excerpt: "[CVPR2021]Refine Myself by Teaching Myself: Feature Refinement: Feature Refinement via Self-Knowledge Distillation"
categories:
  - Paper & Knowledge
  
tags:
  - Paper & Knowledge
 
published: False
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-30T22:19:00-05:00
---

### Abstract
Teacher network를 사용하지 않고 Knowledge Distillation을 적용한다. 즉 Self-Knowledge Distillation을 사용하여 student 네트워크의 성능을 향상시킬 것이다.
이전의 KD 방법들은 classification에 치중되어 <u>local 정보를 활용할 수 없는 방법론이 있었다.</u> 이걸 아주 강조함.

### Introduction & Related Work
이 논문은 KD에서 사용하는 기법들을 크게 3가지로 분류 했다. 
1. class prediction
2. logit (penultimate layer)
3. feature map (sptial information)
abtract에서 저격한 이전 KD 방법론들(local 정보들을 활용하지 않는 방법들)을 더 강조하면서 feature map으로 훈련하지 않는 방법론은 segmentation 같은 task에서 효과가 적다고 말한다. 
또한 self KD를 강조하기 위해 teacher를 훈련하는 것은 손해라고 말함

self-knowledge distillation은 2가지로 나누었다.
1. data augmentation based approach
=> 예를 들어, 같은 이미지에서 augmentation한 이미지를 같게 예측하도록 하게 하는것.
 
2. auxiliary network based approach
=> 중간의 feature들을 auxiliary network로 뽑아서 prediction에 정보를 줌. 

각각의 방법론에는 단점이 존재한다. 
**data augmentation based approach.** local 정보의 손실을 초래한다. 예를 들어서, segmentation task에서 flip한 이미지를 입력으로 넣었다고 해보자. 이 때, 
feature map을 매칭시키는 것을 활용하려 하는데 flip한 이미지로부터 생성된 feature map은 다시 flip한다고 해서 원본을 입력으로 한 feature map이 나오지 않는다. 따라서 local 정보가 중요시 되는 
task에 대하여는 제한적일 수 밖에없다.

**auxiliary network based approach** 따라서 classification만 수행하면 몰라도 더 다양한 task들 해결하기 위해 저자들은 이 방법론을 기반으로 발전시켰다. 하지만 이 방법의 문제점은 상대적으로 크기가 작은 
auxiliary network를 사용하다 보니 teacher와 달리 충분히 refined된 정보를 줄 수 없다. 하지만 이 논문에서는 auxiliary self-teacher network를 제시하여 refined feature map을 생성한다. 

Auxiliary self-teacher network를 만들 때 (정확히는 만드는 것이 아니고 distillation하는 과정을 말한다.) feature network로 top-down을 사용한 FPN이후에 더 발전된 BiFPN을 사용한다.
![](/assets/images/2021-08-30-refine_myself_by_teaching_myself/1.PNG)


### Methods
##### Self-Teacher Network
이 섹션에서는 저자들이 제시한 self-teacher network의 구조를 소개한다. 
![](/assets/images/2021-08-30-refine_myself_by_teaching_myself/2.PNG)
**목표: Introduction과 Related Work에서 언급하였던 self distillation 기법 2가지의 단점을 보완하기 위해 self-teacher network로 refined feature map과 soft label을 활용한다.**
그림을 볼 때 가장 먼저 수행하는 연산은 F이다. F는 가장 기본적인 feature map이고, 그 다음으로는 L, 즉, Lateral convolution 연산이 들어간다. 
* Lateral Convolution Layer
FPN (Feature Pyramid Network)에서 top-down, bottom-up과 lateral connection을 제시했다. 자세한 것은 아래 세 그림을 참고하자. 
![](/assets/images/2021-08-30-refine_myself_by_teaching_myself/3.PNG)
![](/assets/images/2021-08-30-refine_myself_by_teaching_myself/4.PNG)
![](/assets/images/2021-08-30-refine_myself_by_teaching_myself/5.PNG)
이렇게 FPN에서 고안한 multi-scale detection을 통하여 self-teacher network의 윤곽을 잡았다. FPN과 다른것은 FPN은 lateral 연산을 가장 마지막에 하는데 여기서는 가장 먼저한다. 또한 연산량을 줄이기 위해
P1, P4자리에 아무것도없고, L1에서 T1으로, L4에서 T4로 다이렉트로 가는 것을 볼 수 있다. 이 과정에서 P4에서 P3로 top-down되는것을 보완하기 위해 L4에서 P3로 forwarding되고, 
마찬가지로 L1에서 P2로 대각선으로 forwarding된다. 더 자세한 모델 구조는 논문 보면 될듯. 

##### Self-Feature Distillation
이 섹션에서는 전체적인 로스를 설명한다. 

위에서 설명한 self-teacher network의 목표는 refined feature map을 student에 전달하는 것이었다. 중요한 것은 student보다 커지면 안된다는 것. 그렇다면 현재 위의 메인 모델 구조에서 refined feature map은 무엇일까?
직관적으로 보더라도 L과 P 레이어들은 중간 레이어임 알 수 있고, 가장 마지막에 bottom-up 레이어가 teacher network임을 파악할 수 있다. 즉 T레이어들이 refined feature map이다. 또한 이 self-teacher network의 
soft-label도 사용할 것이므로 주의하자. 

추가된 손실함수는 그림에서 볼 수 있듯 3가지이다. 
1. student의 feature과 self-teacher network의 

 





