---
title:  "cv2.gaussianblur 계산법"
excerpt: "cv2.gaussianblur 계산법"
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

커널의 값은 아래의 식을 참고 하면 된다.
![](/assets/images/2021-07-06-gaussianblur/1.JPG)
만약 함수 시그마 값에 0이나 음수를 넣을 경우 시그마 공식은 커널 사이즈를 k라 할때

$0.3 * (0.5 * (k-1) - 1) + 0.8$ 이다.

 

그런데 위에 식에서 앞의 분수가 필요없는 이유는 가우시안 블러 커널이 커널의 모든 원소의 합은 1이 되어야 한다. 
즉, 상수로 스케일링 되는데 어차피 위의 식에서 앞의 분수는 상수이므로 계산 안하고 비율만 계산 하면 된다.