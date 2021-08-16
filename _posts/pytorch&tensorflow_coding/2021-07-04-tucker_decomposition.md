---
title:  "Tucker Decomposition"
excerpt: "Tucker Decomposition에서 Eigenvector 구하기"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch & Tensorflow & Coding
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

![](/assets/images/2021-07-04-tucker_decomposition/1.JPG)
먼저 위와같이 분리를 하고, matmul을 하기 위해 (80x80x3)의 core를 (3x80x80)으로 바꾼다.

![](/assets/images/2021-07-04-tucker_decomposition/2.JPG)
core에 outer1과 outer2를 텐서곱할건데, 먼저 처음에는 matmul로 간단하게 하고, 두번째부터 tensordot을 이용하여 (3x140x80)과 (140x80)을 곱해준다. 
그러면 qwe = (3x140x140)이 되고, 시각화하기 위해서 다시 전치시켜준다. 

그러면 채널별로 아래와 같게 나온다. 

![](/assets/images/2021-07-04-tucker_decomposition/3.JPG)
여기에 마지막으로 outer3를 텐서곱 하면 원래의 이미지가 된다.