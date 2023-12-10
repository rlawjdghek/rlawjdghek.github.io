---
title:  "[CVPR2020]Embedding Expansion: Augmentation in Embedding Space for Deep Metric Learning"
excerpt: "[CVPR2020]Embedding Expansion: Augmentation in Embedding Space for Deep Metric Learning"
categories:
  - Paper & Knowledge
  
tags:
  - Representation Learning 
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-10-09T15:04:00-05:00
---

CVPR 2022를 준비하면서 읽은 논문. 방법론도 좋지만 간단한 방법을 논문화 하기위해 실험이 좋은것 같다. 직관적이고 충분히 추론할 수 있는 확장성 있는 실험으로 저자들이 제안한 방법론을 더 돋보이게 해준다. 

# Abstract & Introduction 
Metric learning의 확장된 분야로 loss를 augmentation과 generalization해주는 방법을 제시한다. 개인적인 생각으로는 metric learning에서 완전히 새로운 loss를 제시하는 것은 상당히 어렵기 때문에
기존에 있는 loss에 추가적인 방법을 곁들여 성능을 향상시키는 방법을 채택했다고 본다. 하지만 논문에서 한가지 아쉬운 점은 SOTA를 달성했다고 말하지만 그 기준이 애매하다는 것이다. 
baseline들은 저자들이 제시한 add-on 방법론이 아닌 비교적 옛날에 소개된 방법들이기 때문에 성능차이가 크게 두드러지지 않는다. 하지만 metric learning에서 2개 이상의 data point를 사용하는 loss
에 대하여 적용가능하기 때문에 (논문에서 사용한 base loss는 4개) 실험이 많았기 때문에 표가 부족해 보이지는 않는다.\
Metric learning에서의 성능 증가는 크게 loss와 Mining (=Sampling)으로 나눌 수 있다. Introduction에서는 저자들이 소개한 방법이 Mining 기반이기 때문에 loss에 대한 설명은 굉장히 짧다. 
Mining 전략은 hard sample을 고르는 것이 목적이다. 왜냐하면 쉬운 image pair들은 기존의 유명한 loss들, arcface, triplet 등으로 쉽게 분류가 가능하기 때문이다. 
**하지만 원초적인 문제는 이렇게 hear sample에만 집중하여 sampling 하다보면, 소수의 hard sample에 모델이 편향이 될 수 있다는 것이다.** 따라서 기존 embedding augmentaion은 
GAN이나 Autoencoder를 활용하였으나 이는 시간과 메모리를 많이 잡아먹기 때문에 효율성이 떨어진다. 따라서 독립적인 모델을 제시하기 보다 간단하고 모델과 함께 
end-to-end로 학습 가능한 방법이 필요하다. 

# Methods
저자들이 제시한 방법론은 크게 2가지라 할 수 있다.
1. 2개의 클래스에서 각 클래스에 속하는 2개의 data point로부터 interpolation을 구한뒤, normalization하여 hypersphere에 projection
2. projection된 서로 다른 클래스에서 가장 짧은 distance를 가지는 점, hard negative point 선택. 

Triplet loss, lifted loss, N-Pair loss, MS loss 총 4개의 loss에 대하여 적용한다. 식은 기존의 식과 크게 다르지 않다.

방법론이 간단하므로 아래 그림을 보면 이해할 수 있다. 
![](/assets/images/2021-10-09-Embedding_Expansion/1.PNG)
![](/assets/images/2021-10-09-Embedding_Expansion/2.PNG)
  

# Experiment
### 1. 4개의 로스에 대하여 각각 EE를 붙인 것과 안붙인 것의 성능을 보여줌. NMI와 F1, Recall을 사용함. \ 

* NMI를 예제를 통해서 계산해보자. \
$Y = class labels$ \ 
$C = cluster labels$ \
$H() = entropy$ \
$I(Y; C) = Mutual\;\;Information$ (log들은 2를 밑으로 가진다.)\
$NMI(Y, C) = \frac{2 \times I(Y;C)}{\[H(Y) + H(C)\]}$ \
![](/assets/images/2021-10-09-Embedding_Expansion/8.PNG)의 NMI를 구해보자.
위의 상황에서는 4가지만 구하면 된다.

$H(Y|C=1) = -P(C=1) \sum P(Y=y | C=1) log(P(Y=y | C=1))$ \
$H(Y|C=2)$ \
$H(Y)$ \
$H(C)$ 


복잡해지므로 $H(Y\|C=1)$ 만 구해보면, 
$P(Y=1 | C=1) = \frac{3}{10}$, $P(Y=2 | C=1) = \frac{7}{10}$, $P(Y=3 | C=1) = \frac{0}{10}$이고,
$H(Y | C=1) = -P(C=1) \sum_{y \in \{1, 2, 3\}} P(Y=y | C=1)log(P(Y=y | C=1)) $ \
$ = - \frac{1}{2} \times \[\frac{3}{10}\ log(\frac{3}{10}) + \frac{0}{10}log(\frac{0}{10}) + \frac{7}{10}log(\frac{7}{10})] = 0.4406$
이런 식으로 $H(Y|C=2)$를 구하고, entropy 구하는 식으로 $H(Y), H(C)$를 구하면 된다. $I(Y;C) = H(Y) - H(Y|C)$이고 대입.

결과표는 논문 마지막에 있음. 

### 2. Labels of Synthetic Points
Synthetic point 사용의 장점은 이 점들은 이미 같은 클러스터에 있는 두점 사이의 보간된 값이기 때문에 그 클러스터에 들어갈 확률이 더 높다. 논문에서는 high degree라고 표현하는데, 이는 더 쉽게 분류될 수 있다는 것을 
의미한다. 아래 그림은 synthetic data과 original data의 recall을 보여준다. synthetic이 더 높은것을 알 수 있다.  
![](/assets/images/2021-10-09-Embedding_Expansion/3.PNG) 

### 3. Impact of $L_2$ normalization
interpolation point들을 hypersphere에 projection하는 효과를 보여줌. 
![](/assets/images/2021-10-09-Embedding_Expansion/9.PNG)

### 4. Impact of number of synthetic points
적절한 synthetic point의 갯수를 보여준다. 2~8이 적당함. 왜 많이 쓰면 안좋아지는지 모르겟다.
![](/assets/images/2021-10-09-Embedding_Expansion/4.PNG)

### 5. Selection Ratio of Synthetic Points
Interpolation point들이 hard negative sample로써 얼마나 선택되는지 보여줌. 훈련될 수록 cluster가 잘 형성된다. 따라서 interpolation된 데이터는 같은 클래스에 속하는
두 샘플 사잇값이므로 점점 더 쉬운 데이터가 된다. 기하학적으로 생각해봐도 가장 edge에 있는 두 샘플이 다른 클러스터에 포함되는 point와 가장 가깝다는 것을 알 수 있다.
![](/assets/images/2021-10-09-Embedding_Expansion/5.PNG)
위의 그림에서 epoch이 지날수록 synthetic data가 hard negative로 선택되는 비율이 낮아지는 것을 볼 수 있다. 

### 6. Effect of hard negative pair mining
synthetic data가 epoch가 지날수록 더 hard sample이 되기는 어렵지만, 그래도 hard sample로 존재는 한다는 것을 알 수 있다. 이는 곧 기존 original point만 사용하는 것 보다 더 많은
hard sample을 보유하고 있는 것이므로 이를 보여주기 위해 epoch이 지날수록 각 hard negative sample의 거리를 보여주었다.
![](/assets/images/2021-10-09-Embedding_Expansion/6.PNG)
위의 그림을 보면, 다른 클러스터끼리는 거리가 멀어야 하므로 파란색, 대각선은 빨간색이 나올수록 잘 분류된 것을 말한다. triplet loss만 사용한 모델은 너무 잘 분류하여 대각전 이외의 부분이 파란색으로 되어
더이상 hard sample이 존재하지 않는다는 것을 보여준다. 반면, EE를 적용한 것은 노란색이 전체적으로 더 분포하는 것을 보여준다. 따라서 훈련이 진행되도 hard sample이 여전히 존재한다는 것을 알 수 있다. 

### 7. Robustness
더 hard한 negative sample을 사용함으로써 모델의 robustness가 증가했다는 것을 보여준다. Occlusion을 주고 그에 따른 성능을 그래프로 표현하였다. 
![](/assets/images/2021-10-09-Embedding_Expansion/10.PNG) 

### 8. Training speed and memory
훈련 속도와 메모리 사용량을 비교한다. 단순 선형 보간 식이기 때문에 추가적인 자원 소모는 거의없다. 
![](/assets/images/2021-10-09-Embedding_Expansion/7.PNG)

























