---
title:  "Reducing the Teacher-Student Gap via Spherical Knowledge Distillation"
excerpt: "Reducing the Teacher-Student Gap via Spherical Knowledge Distillation"
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

D스터디 4주차 때 한 논문인데 아직 억셉 안된 논문. 이것도 쉽다


### Abstract & Introduction

별 내용 없고 저번 논문과 같이 teacher와 student의 격차가 커질수록 성능이 저하되는것을 막기 위한 방법을 소개한다. 기존의 문제점은 결국 모델이 원하는 것은 작은 로스, 즉 정확도가 같아도 confidence차이로 더 작은 로스를 만들수 있다. 모델이 커지면 정확하게 분류하는 것은 더 정확하게 분류하는 성질이 있다. 따라서 극명하게 차이나는 confidence가 KD의 학습에 방해가 될 수 있기 때문에 격차를 줄이기 위한 방법을 고안해 낸다. 또한 KD의 고질적인 문제중 하나는 temperature에 민감하다는 것인데 저자들이 제시한 방법은 temperature에 크게 영향을 받지 않는다.

세가지 contribution

 

1. capacity gap을 완화시킨다.

2. easy to opimize

3. temperature에 robust하다. 


### Spherical Knowledge Distillation

아직 accept가 안됬고 supplementary자료도 없어 증명이 완벽하다고는 할 수 없다. 개인적으로는 equation (4)가 틀렸다고 생각한다. 
또한 단순히 기존의 logit (softmax를 거치지 않은 생 logit)을 norm과 norm으로 나눈 것으로 분리시키고 증명을 하는 것 자체에서 맞는 접근인지 의구심이 든다. 결국은 아래 그림을 
해주는 것이 다다. 기존 KD는 그냥 소프트맥스를 취한 값을 soft label로 취급하여 student 모델을 학습시켰다면 SKD는 기존의 logit에서 student와 teacherd의 logit 전체의
 값들을 RMSE을 한 값과, student 값만 normalization 하고, teacher값만 normalization 한 다음 방금전에 전체 값을 RMSE를 한 값을 곱해주면 조금이나마 값이 달라진다. 
 이것을 소프트 맥스를 취한다. 즉, 기존에 너무 극명하게 나뉘었던 logit값을 소프트맥스를 취하면 여전히 극명하므로 이를 개선시킨 점이다. temperature의 변수를 사용하지 않고 
 자기 자신의 정보를 이용하여 개선 시킨 것이다. 개인적인 생각으로 이 부분은 맞다고 생각하는데 여전히 capacity gap을 gradient로 설명하는 부분이 이해가 되지 않는다. 
  확실히 temperature에 robust한 것은 맞다고 생각한다. 그러면 또 하나 의구심이 드는것이 만약 temperature에 robust한 것이라면 결국 temperature와 같은 기능을 하는 방법론을 
  제시한 것은 아닌가? capacity gap을 줄이는 것이 아니라 최적의 temperature를 찾는 방법에 가까운 것이 아닐까 하는 추측이 든다.
  
![](/assets/images/2021-07-19-Spherical_KD/1.JPG)
  
  