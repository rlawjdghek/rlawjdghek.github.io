---
title:  "SphereFace2: Binary Classificarion is All You Need for Deep Face Recognition"
excerpt: "SphereFace2: Binary Classificarion is All You Need for Deep Face Recognition"
categories:
  - Paper & Knowledge
  
tags:
  - Representation Learning 
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-10-27T15:04:00-05:00
---

Sphereface 저자가 쓴 논문. 2가 붙은 것은 Sphereface의 후속이 아니라 binary classification을 의미한다고 하지만 메인 아이디어는 sphereface와 연관이 깊어 보인다.
지금까지 facial representation learning에서 축을 이루는 개념들을 (arcface, cosface) 정리하고 묶은 논문이라 볼 수 있다. Facial recognition에서 현재 가장 좋은 
성능을 보인다. 

# Abstract & Introduction
기존의 multi-class classification은 softmax (cross entropy) loss에 기반하여 쉽게 over-confident한 문제를 보인다. 따라서 수십만개의 identity가 있는 facial 
recognition class에서 일반화가 부족하다. 저자는 지금까지의 representation learning에서 제안된 방법론들을 Triplet과 pair-base로 한번 나누었고, Proxy의 유무로 다시 한번 나누었다. 
예를 들어 arcface, cosface, sphereface는 proxy를 사용하고 negative와 positive를 모두 이용하므로 triplet이라 할 수 있다. \ 
**triplet loss (max(0, D(anchor, positive) - D(anchor, negative) + margin)) 과 다르다는 것을 유의하자.**\ 
이외의 설명들은 모두 section 3부터 자세히 나와있으므로 contribution만 정리하자. 

1. 기존과 다르게 positive인 같은 클래스와 나머지를 모두 negative로 다루는 이진분류를 활용하였다. 이는 병렬처리와 noise에서 더 좋은 성능을 발휘한다.
2. 지금까지 representation learning에서 제안된 method들을 하나의 흐름으로 정리하면서 성능 향상에 도움이 되는 방법론들을 설명한다.
3. 이 논문을 기반으로 여러 새로운 아이디어들을 가지뻗기 식으로 개발할 수 있다. 

# SphereFace2 Framework
개인적으로 sphereface2는 여러 방법론을 합친것을 기반으로 새로운 아이디어를 붙였다고 볼 수 있다. 학습에 도움이 되는 아이디어를 하나씩 설명한다.
### Positive / negative sample balance
첫번째로 고려해야 할 것은 positive sample과 negative sample의 균형이다. identity가 굉장히 많기 때문에 절대적으로 negative sample이 많다.
이를 해결하기 위해 제안된 방법은 negative term에 penalty 계수를 붙이는 것이다. 따라서 아래와 같은 식을 얻을 수 있다. 

$L_b = \lambda log(1 + exp(-cos(\theta_y))) + (1 - \lambda)\sum_{i \neq y} log(1 + exp(cos(\theta_i)))$

이 때 $\lambda = \frac{K-1}{K}, (K = number of identity)$. 

### Easy / Hard sample mining
수 많은 샘플들 중에는 쉬운 샘플이 있고, 어려운 샘플이 있다. 두 번째로 고려해야 할 것은 이 샘플들 간의 가중치다. sphereface에서 제시한 scale factor s는 코사인 값에 곱해진다. 
하지만 s의 값에 따라서 hard와 easy sample의 가중치가 달라지는데, s를 사용하지 않기에는 s가 가진 장점을 포기할 수 없었나 하는 생각이 든다. 
![](/assets/images/2021-10-27-SphereFace2/1.JPG)
위의 그림에서 hard sample과 easy sample간에 로스를 본다. 당연히 hard sample에서는 손실 함수의 값이 커지는데 중요한 것은 곡선의 기울기이다. 
왼쪽 그림은 sphereface에서 제시한 s-normalized softmax loss를 사용하였다. hard sample에서 급격하게 손실이 증가하는 것을 보면, hard에 많은 가중치가 있다는 것을 알 수 있다.
즉, easy sample보다 hard sample에 학습이 더 민감해지기 때문에 balance가 깨질 수 있다. \
따라서 우리는 이 scale s에 다시 한번 penalty를 주어야 한다. 저자들은 아래 로스와 같이 log의 값에 $\frac{1}{r}$을 곱해주었다.

$L_e = \frac{\lambda}{r} log(1 + exp(-r \cdot cos(\theta_y))) + \frac{1 - \lambda}{r}\sum_{i \neq y} log(1 + exp(r \cdot cos(\theta_i)))$

### Angular margin
Cosface와 Arcface에서 소개한 angular margin은 굉장히 중요하다. 따라서 저자는 최종 손실 함수에 이 개념도 추가하였다. arcface대신 cosface를 사용하였는데 논문에서는 
아무거나 사용해도 상관 없지만 둘 다 사용해서 비교하는것은 exhaustive하기 떄문에 cosface만 적용하였다고 한다. 따라서 $L_a$의 식은 $L_e$ 식에서 cosface를 차용하였다. 하지만 표기적으로 다른점은 positive와 
negative에 같은 값을 부호만 바꿔서 적용하였다는 것. 사실상 cosface에서는 positive항에 대해서만 margin을 주었는데 이는 positive와 negative에 $\frac{m}{2}$씩 margin을 주는 것과 같다.

$L_a = \frac{\lambda}{r} log(1 + exp(-r \cdot (cos(\theta_y) - m_p))) + \frac{1 - \lambda}{r}\sum_{i \neq y} log(1 + exp(r \cdot (cos(\theta_i) + m_n)))$

추가적으로, bias를 강조하였다. $L_a$에 decision boundary를 나타내는 bias를 붙여 성능을 더욱 올릴 수 있다고 주장한다. bias가 의미하는 바는 아래 그림과 같다.
![](/assets/images/2021-10-27-SphereFace2/2.JPG) 
두번째와 세번째 그림을 보면, 두번쨰 그림은 bias가 없기 때문에 decision boundary가 정중앙을 지나면서 class 1과 나머지 클래스들을 구분한다. 이것은 2차원상에 그렸기 때문에 
다른 클래스들에 대하여 일반화가 안되었다고 생각될 수 있지만, 고차원에서는 가능하다. 반면 세번째 그림과 같이 bias를 붙이게 되면 decision boundary가 평행이동 하게 되어 hypersphere상에 있는
feature embedding vector들에 더 가까이 있을 수 있게 된다. bias를 적용한 loss는 아래와 같다. 

$L_a = \frac{\lambda}{r} log(1 + exp(-r \cdot (cos(\theta_y) - m_p) - b)) + \frac{1 - \lambda}{r}\sum_{i \neq y} log(1 + exp(r \cdot (cos(\theta_i) + m_n) + b))$

한가지 유의할 점은 식에서는 b_y, b_i로 했지만 실제로는 같은 값을 가리킨다. 왜냐하면 open-set에 대해서는 특정한 클래스가 의미가 없으므로 같은 값을 사용해도 무방하다. 또한 margin도 마찬가지이다. 

### Similarity adjustment
마지막으로 positive와 negative들의 simiarity를 비교 했더니 아래 그림과 같이 negative의 분산이 더 높았다. 또한 simiarity의 범위도 굉장히 imbalance하다. 
![](/assets/images/2021-10-27-SphereFace2/3.JPG)
이러한 통찰로 인하여 저자들은 새로운 함수를 고안하였다. 코사인 값을 어떤 함수를 통과시켜 코사인 분포 자체를 변화시키려 한다. 

$g(z) = 2 (\frac{z+1}{2})^t - 1, z \in \[-1, 1\]$

z에 들어가는 것은 $cos(\theta)$이고, 이는 $g(cos(\theta)) \in \[-1, 1\]$이다. 중요한 하이퍼파라미터 t가 분포의 범위를 조절한다. 함수 g를 t값에 대하여 비교하여 보면 아래 그림과 같다.

![](/assets/images/2021-10-27-SphereFace2/4.JPG)

t가 증가할수록 경사가 심해지는 것을 생각하면, 위의 코사인 분포 그림 두번째, 세번째 그림과 같이 분포가 더 나누어지는 것을 볼 수 있다. 또한 negative sample 분포의 분산이 더 낮아지는 것을 볼 수 있다.

### Fianl loss function
함수 g를 마지막으로 소개된 모든 것을 종합하면 최종 로스가 완성된다.
$L_a = \frac{\lambda}{r} log(1 + exp(-r \cdot (g(cos(\theta_y)) - m_p) - b)) + \frac{1 - \lambda}{r}\sum_{i \neq y} log(1 + exp(r \cdot (g(cos(\theta_i)) + m_n) + b))$

#Geometric Interpretation
이제 위의 최종 손실 함수에서 4개의 하이퍼파라미터에 대한 기하학적인 설명을 덧붙인다.
1. r: hypersphere의 반지름, feature의 magnitude
2. b: baseline decision boundary
3. m: angular margin
4. t: cosine distribution
아래의 그림이 한번에 설명해준다. 각 변수 기호들이 나타낸 것에 주의하여 정리해보자.
![](/assets/images/2021-10-27-SphereFace2/5.JPG)

# Experiment
sphereface2는 좋은 결과를 내므로 결과 테이블은 생략한다. 대신 하이퍼파라미터 세팅만 정리해두자.
1. r과 m은 기존의 연구에서 많이 개선되어 왔으므로 보편적인 값을 따라간다. $r=30, m=0.4$
2. $\lambda, t$는 ablation을 통해 구해낸다. $\lambda=0.6, 0.7$, $t = 3, 5$


