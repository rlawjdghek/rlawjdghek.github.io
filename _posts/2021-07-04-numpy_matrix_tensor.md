---
title:  "넘파이 행렬, tensor 곱"
excerpt: "넘파이 차원별로 정리"
categories:
  - Numpy
  
tags:
  - Numpy
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

### 2차원
np.matmul과 np.dot이 같다. 차원은 (A,B) == (row, col) 로 바라본다. 

### 3차원 텐서 표현
3차원에서는 넘파이 텐서 (A,B,C)를 (채널, row, col)로 바라본다.

예를 들어

np.arange(24).reshape(2,3,4) 텐서가 있으면 depth의 개념이 2인것. 따라서 np.arange(15).reshape(5,3)와 np.matmul
하면 (2,5,4)의 행렬이 된다. 넘파이 텐서를 출력할때의 배열을 종이에 표기할 때에는 축이 달라지는 것을 주의하자. 수학에서는 
(A,B,C)의 3차원 텐서에서 A는 row, B는 col, C는 depth라고 생각하기 때문에 아래의 그림을 참고하여 그린다.

![](/assets/images/2021-07-04-numpy_matrix_tensor/1.JPG)

즉, 만약 종이에 수학적으로 차원이(i,j,k)를 써서 넘파이로 적용시키려고 하면, (k, i, j)로 reshape하면 된다.
예를 들어 실제 수학에서 0~23까지의 숫자를 (2,3,4)차원으로 만드려고 할때에는 아래 그림과 같이 np.arange(24).reshape(4,2,3)하면 된다.
![](/assets/images/2021-07-04-numpy_matrix_tensor/2.JPG)

### 3차원 텐서곱
이제 3차원에서의 텐서곱을 다뤄보자. np.tensordot은 축 개념 때문에 상당히 복잡하다. 아래 코드를 보면서 규칙을 외우듯이 배우자.

![](/assets/images/2021-07-04-numpy_matrix_tensor/3.JPG)
수학에서처럼 행렬x텐서를 하기 위해서 첫번째 변수가 행렬이 오는 것이 아니라 텐서가 먼저 와야 원하는 결과를 얻을 수 있다. 
axes변수가 어려운데 행렬과 텐서의 slice를 행렬과 곱하기 위해 서로 대응되는 차원을 쓴다.
**그러고나면 계산 순서는 무조건 축을 기준으로 한다. 아래 그림을 보면 행렬에서는 차원 1에 해당하는 col을 기준으로 움직이고
텐서에서는 차원 1에 해당하는 row를 기준으로 움직이다. 아래 그림은 위의 그림의 2번째 셀의 첫번째 슬라이스 결과에 해당한다.
행렬곱으로 바라보는 것이 아닌 텐서곱으로 바라보자. 매개변수의 순서가 수학과 달라도 축이 같으면 순서에 상관없이 계산한다. 텐서가 먼저 들어갔기 때문에
transpose해서 원하는 모양 맞춰주기.**
![](/assets/images/2021-07-04-numpy_matrix_tensor/4.JPG)





