---
title:  "[NIPS2018]Knowledge Distillation by On-the-Fly Native Ensemble"
excerpt: "[NIPS2018]Knowledge Distillation by On-the-Fly Native Ensemble"
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

### Abstract & Introduction
이것도 self-distillation 논문이다. 문일철 교수님의 FRSKD와 마찬가지로 self-distillation의 출현의 주 목적인 teacher의 사용을 비판하면서 
시작한다 (i.e., longer training time...). 그 다음 이 논문에서는 self-distillation을 online learning을 중점으로 접근한다. Be your own teacher와 같이 
auxiliary network를 사용하는 방법론도 모두 online learning 이라 할 수 있다. (이 논문에서는 offline learning을 teacher를 고정시켜두는 것만 예시로 듦. 개인적인 
생각으로는 self-distillation에서 data augmentation 방법론도 offline이라 할 수 있을듯 하다.) 
 
논문에서 주장하는 기존의 online learning은 peer-teaching manner, 즉 student를 여러개 두고 한 개의 student를 학습하는 것이다. 이러한 기존의 학습 방법은 아래와 같이
3가지의 단점을 가진다. 
1. peer-student는 limited information을 준다. 즉, capacity가 작아서 같은 student를 훈련할 때 의미있는 지식을 줄 수 없다. 
2. 여러개의 student를 사용하는 것이 teacher 보다 더 큰 자원 소모를 할 수 있다. 
3. 동시다발적인 훈련이 복잡해져서 역전파 과정에서 정확한 알고리즘을 구현 할 수가 없다.     

따라서 이 논문에서는 여러개의 student 모델을 사용하지 않고 branch network를 활용하여 한개의 모델을 구성한다. 또한 제목의 on-the-fly는 "즉석"이라는 뜻으로 하나의 모델 훈련과정에서
앙상블 모델이 훈련되고 그것이 teacher가 된다는 것이다. 

### Method
![](/assets/images/2021-09-03-on_the_fly_KD/1.PNG)
이 논문의 방법론은 꽤나 쉬워서 그림으로도 충분히 이해가 가능하다. 

일반화를 위해 branch network를 m개를 두었는데 논문에서는 최대 5개까지 사용하였다. 또한 5개가 넘어가면 그냥 teacher를 사용하는 것과 같다고 생각이든다. 
여기서의 teacher network는 그림에서 branch network들이라고 생각하면 된다. 이 branch network들은 같은 low-level feature extractor를 공유하고,
이 network들로부터 앙상블 한 결과가 좋을 것이므로 그것이 의미있는 logit이 되고 결국 teacher model이 되는 것이다.
한가지 주목할 점은 중간에 low-level에서 위로 뽑아낸 gate layer인데 이는 나중에 가중치처럼 각 branch network마지막에 곱해진다.  
나머지 부분에서는 새로운 아이디어를 사용하지 않았다. cross entropy와 KL divergence로스만 사용하였음.

따라서 아래 알고리즘을 보면, 
![](/assets/images/2021-09-03-on_the_fly_KD/2.PNG)
최종 student는 첫번째 branch를 사용하였고, 만약 더 좋은 결과를 얻고 싶다면 여러개의 branch network를 사용하면 된다.

### Experiment
실험 데이터셋은 단순한 classification이다. 특별한 것은 없음. Hinton의 KD와 Deep mutual learning 모델과 비교해서 제일 잘 나왔고, Hinton은 teacher를 사용하였음에도 ONE보다 성능이
낮다는 것이 주목할만하다. 그런데 이 당시 최신 기법들과 비교하면 좋지는 않은듯. 

그 뒤로 앙상블 한 결과를 보여주고 (당연히 앙상블 하면 잘 나옴), online learning을 사용/미사용, low-level feature extractor를 사용/미사용, gating layer를 사용/미사용한 결과를 보여준다.
![](/assets/images/2021-09-03-on_the_fly_KD/3.PNG)
위의 결과 해석에서 의아한 부분이 있는데 ONE의 결과에서 (앙상블 사용 x) low-level feature를 공유한 것보다 공유하지 않은 것의 성능이 더 잘 나온다. <u>(사실 이 부분만 
low-level을 공유하지 않은 앙상블이 성능이 더 낮다고 바뀌면 모두 말이 된다.)</u>즉, 공유 한것이 더 좋은 teacher라는 말인데, 
이러면 더 좋은 teacher이면 성능도 더 잘 나와야 한다는 결론이 된다. 하지만 teacher (앙상블 사용)한 결과를 비교하면 공유한 것이 더 낮다. 모순되는 결과. 뒤의 4.6 섹션에서도 앙상블 teacher에 대한 분산을 계산한다.
3개의 peer를 앙상블 한 것이 3개의 branch(low-level 공유)보다 분산이 더 높다고 주장. 즉, 3개의 peer를 앙상블 한 것이 generalise 성능이 떨어진다. 이 말은, 새로운 데이터를 가져 왔을 경우 
성능이 더 낮다는 말인데 앞에서 나온 결과와 모순된다. 저자가 마지막으로 추측하는 것은,
**mean generalisation capability 측면에서 branch를 사용한 것의 성능이 잘 나온 것을 독립성을 좀 더 줄이고 모델의 capacity를 키우는 것으로 이해한다면 오히려 큰 모델이 작은 것을 여러개 합친 것보다 더 나을
 수도 있다.**
 

  


