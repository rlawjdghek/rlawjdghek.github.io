---
title:  "pytorch model.train(), eval()"
excerpt: "pytorch model.train(), eval()"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch & Tensorflow & Coding
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-05T12:48:00-05:00
---

model.train() 하면 drop out과 BN이 적용이 되는데 DROPOUT은 예를 들어 p=0.4라고 하면 40퍼센트를 사용하지 않겠다는 말이다. 마지막 결과에 3/5을 곱하여 가중치를 더 죽인다고 생각. 
그래서 훈련 한 뒤에 eval을 쓰면 모든 가중치를 다 살려내므로 정상적인 1의 결과가 나올 수 있다. 

BN도 비슷하게 레이어마다의 배치 정규화를 진행 하면서 mean variance를 모아두었다가 eval에서 모아둔 값들을 평균계산해서 적용.