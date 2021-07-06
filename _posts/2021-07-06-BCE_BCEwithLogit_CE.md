---
title:  "BCE & BCEwithLogit & CE"
excerpt: "BCE & BCEwithLogit & CE"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T14:48:00-05:00
---

맨날 BCE만 하다보니까 마지막 fc 레이어 다음에 sigmoid 를 쓰는게 습관이 되서 적는다. 

multi class를 할때에는 CE를 적는데 BCE는 sigmoid가 적용이 안되어있고, BCE(output, label)에서 output에는 
무조건 0과 1 사이의 확률값이 나와야 되기 때문에 sigmoid를 거쳐야 하는 것이다. 따라서 BCE는 output이 1개가 나와도 되는 것이고, 

CE는 자동으로 안에 softmax 함수가 있다. 그래서 그냥 fc만 거친 값, 즉 음수가 될 수 있는 값들이 들어가기 때문에 따로 sigmoid나 softmax를 안해도 된다. 


BCELosswithLogit 은 sigmoid를 포함하고 있기 때문에 생 logit값을 넣어야 한다. 