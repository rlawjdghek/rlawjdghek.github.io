---
title:  "[CVPR2021]Revisiting Knowledge Distillation: An Inheritance and Exploration Framework"
excerpt: "[CVPR2021]Revisiting Knowledge Distillation: An Inheritance and Exploration Framework"
categories:
  - Paper & Knowledge
  
tags:
  - Knowledge Distillation
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-09-29T14:33:00-05:00
---
방법론은 아주 간단하지만, 실험이나 자신의 방법에 대한 증명이 매우 깔끔하다. 또한 밑바닥부터 아이디어를 낸 것이 아니라, 기존의 방법들과 결합 할 수 잇는
방법을 제시함으로써 에드온하여 성능을 높였다. 나중에 논문 쓸 때 실험 파트 참고하면 좋을듯.

우선 아래 그림으로 전체적인 내용을 빠르게 이해할 수 있다. (a)의 위 그림에서 teacher는 치타이미지를 악어롤 분류하였다. 그와 비슷하게 teacher의 지식을 기반으로 학습한 student또한
치타를 악어로 분류하였고, activation map이 teacher와 아주 유사하다.\
그 다음 아래 그림에서 또한 student가 teacher를 따라가도록 하는 inheritance part는 당연히 teacher를 따라간다. 하지만 teacher의 반대 방향으로 penalty를 주는 exploration part는
치타의 머리를 더 강조함으로써 student모델에 영향을 주었다. 따라서 저자들이 제시한 새로운 방법론으로 학습된 student 모델은 치타로 옳게 분류 할 수 있다. 

또한 (b)의 그림은 student가 teacher보다 클 때, 저자의 방법 > 일반적인 학습 > KD순서로 잘 나온 결과를 보여준다. (진한 곡선 참고.)\
즉, 더 안좋은 모델로 좋은 모델을 훈련 시킬경우 그대로 따라하는 것은 일반적인 방법보다 안좋다는 것을 입증하고, 더 않좋은 모델로부터 penalty를 받은 좋은 모델이 가장 좋은 결과를 보여준다.

한가지 질문하고 싶은 것은: representation이 부족하다는 것. capacity가 충분하고, active neuron 갯수가 비슷한데 왜 teacher의 성능을 못 내는 student가 있는지, 페널티를 준다면
적어도 teacher는 완벽히 따라하고 주어야 하는 것 아닌지.

![](/assets/images/2021-09-29-Inheritance&Exploration_KD/1.PNG)
### Abstract & Introduction
지금까지의 KD방법들은 모두 teacher network를 따라가는데에 급급하였다. 하지만 teacher는 optimal한 정보를 전달하지 않으므로 student의 입장에서
teacher가 주는 대로 학습하는 것은 분명 한계가 존재한다. 이 논문에서는 기존과 같이 teacher를 따라가는 특성 inheritance와 teacher를 벗어나 더 창의성을 추구하는 
exploration을 소개한다. 논문의 표현을 그대로 적자면,

"**The inheritance part** is learnd with a similarity loss to transfer the existing learnd knowledge from the teacher model to the student model," \
"while **the exploration part** is encouraged to learn representations different from the inherited ones with a dis-similarity loss."

이 두 방법을 결합함으로써, 저자들은 teacher의 정보보다 더 정확한 정보를 전달 할 수 있다고 주장한다. \
이것이 가능한 이유는 student의 capacity가 이미 충분하다는 가정이 있다. 즉, student는 데이터셋을 효용하는 크기를 갖고 있지만, representation이 약하기 때문에 성능이 
안나온다는 연구결과를 참조한다. ([NIPS2014]Fitnets: Hints for thin deep nets) \
따라서 모든 parameter를 사용할 필요가 없으므로 저자들은 **기존의 student에서 일부는 teacher를 따라가고, 일부를 이외의 창의적인 표현을 배울 수 있다고 한다.**

### Methods
![](/assets/images/2021-09-29-Inheritance&Exploration_KD/2.PNG)
inheritance / exploration을 하기위해 shared latent space (encoder)를 이용하여 shape를 맞춰준다. 
훈련 framework에서 사용되는 feature는 아래와 같이 표기 할 수 있다. 
1. student의 inheritance: $f_{inh}$
2. student의 exploration: $f_{exp}$
3. teacher의 intermediate feature: $f_T$
각각은 encoder 또는 auto-encoder를 통과하여 $F_{inh}$, $F_{exp}$, $F_T$로 표기된다.
 
#### Compact Knowledge Extraction
우선 auto-encoder를 이용하여 teacher의 logit을 가공한다. 일반적인 reconstruction loss를 사용.
$\mathcal{L}_{rec} = \parallel f_T - R(f_T)\parallel^2$

#### Inheritance and Exploration
우선 각 loss를 적용시킬 intermediate feature map을 랜덤으로 고른다. 실험 파트에서 각각의 비중을 정하는 ablation을 보여준다. 

$\mathcal{L}_{inh} = \parallel \frac{F\_{inh}}{\parallel F\_{inh}\parallel_2}-\frac{F_T}{\parallel F_T\parallel_2}\parallel_1$

$\mathcal{L}\_{exp} = \parallel \frac{F\_{exp}}{\parallel F_{exp}\parallel_2} - \frac{F_T}{\parallel F_T\parallel_2}\parallel_1$

$\mathcal{L}\_{exp}$는 $\mathcal{L}\_{inh}$와 반대되는 손실함수로써 $F_{exp}$를 teacher의 feature를 배우지 않도록 penalty한다. 저자들은 이를 
"encourage the exploration part to focus on other regions of the images, exploring new features that are complementary to the inherited ones."
라고 표현했는데, 단지 반해방향 로스를 주는것만으로도 가능한 것인지는 잘 모르겠다. 

결과적으로 최종 손실함수는 아래와 같다.

$\mathcal{L} = \mathcal{L}\_{goal} + \lambda_{inh}\mathcal{L}\_{inh} + \lambda_{exp}\mathcal{L}\_{exp}$

#### Extension to Deep Mutual Learning
이 논문에서의 장점이라 생각되는 섹션이다. student가 teacher와 반대되는 성질을 배움으로써 그 지식을 통하여 teacher를 훈련 시킬 수 있다. teacher와 student를 번갈아가면서 한번씩 훈련한다. 

### Experiment 
**Method가 간단하지만 많은 창의적인 실험을 통하여 결과 해석을 다채롭게 하였다. 이 논문에서 사용한 방법론을 적용하려 노력해보자.**


#### Image classification & Object detection
![](/assets/images/2021-09-29-Inheritance&Exploration_KD/3.PNG)
먼저 기본적인 분류에서의 성능이다. 이 논문에서 강조하는 것처럼 teacher를 무작정 따라가지 않고 적당한 penalty를 준다면 성능이 상승한다는 것을 보여주었다. 기존의 방법론에 적용을 하여 
성능을 일관성있게 개선하였을 뿐만 아니라, sota를 달성하였다. (하지만 비교군들이 그렇게 최신 방법들은 아니다.)

최신 KD 논문들이 그렇듯 다른 task에서도 성능이 개선된다는 것을 보여주어야 하기 떄문에 object detection에 대해서도 추가적인 실험을 하였음. 

#### Extension to Deep mutual Learning
위에서 언급하였듯 mutual learning에서도 성능향상을 보였다. 


### Ablation Study
Interitance와 Exploration에 대하여 추가적인 분석을 진행하였다.

#### Inheritance
1. Layer-wise relevance propagation (LRP):  heat map 기반 activation 분석.
![](/assets/images/2021-09-29-Inheritance&Exploration_KD/4.PNG)
첫번째 그림과 마찬가지의 해석이 가능하다. 여기서는 더욱 극명한 차이를 보여줌.
2. Number of Active neuron: loss function에 반응한 방향의 수로써 많을 수록 redundancy가 적다고 할 수 있다.
teacher는 46개, student의 inheritance는 45개이지만, 총 neuron 갯수는 teacher가 student의 2배이다. 즉 teacher는 더 많은 neuon을 갖고 있지만, 활용하는 neuron갯수는 
student와 비슷하기 떄문에 이미 student의 capacity는 충분하다고 할 수 있다. 충분하다고 할 수 있다. 

#### Exploration
![](/assets/images/2021-09-29-Inheritance&Exploration_KD/5.PNG)
1. 첫번째, 세번째 그림에서 보여주었듯이 exploration part에서는 teacher, basic KD, inheritance와 다른 부분에 집중하여 효과적으로 새로운 지식을 배울 수 있다.
2. Centerd Kernel Alignment (CKA): representation을 측정하는 척도. 낮을수록 다양한 representation이 있다고 할 수 있다.
위의 그림 (a)를 보면 기존 student보다 더 낮은 것을 볼 수 있다. 
3. Number of Active neuron
위의 그림 (b)를 보면 기존 student보다 더 높은 것을 알 수 있다.










