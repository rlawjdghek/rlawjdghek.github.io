---
title:  "Differential Privacy"
excerpt: "Differential Privacy"
categories:
  - Paper & Knowledge
  
tags:
  - Paper & Knowledge
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-14T15:33:00-05:00
---

![](/assets/images/2021-08-14-Differential_Privacy/1.PNG)


위의 정의는 작은 $\epsilon$에 대해서 공격하는 사람이 두개의 비슷한 dataset을 보고 구별할 수 없다는 것을 의미한다. 

식을 해석해보자면 서로다른 데이터 셋이 있는데 알고리즘을 통과한 결과가 내가 추적하고 싶은 ($\mathcal{S}$) 데이터에 들어가는지 안들어가는지 모르면 DP가 성립한다. 예를들어 
$\epsilon > 0$이므로 만약 완벽하게 숨기는 알고리즘을 생각해 볼 때 Pr의 값은 같을 것이다. 따라서 부등식이 성립한다. 