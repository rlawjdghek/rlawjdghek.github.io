---
title:  "[ICCV2019]Be Your Own Teacher: Imporve the Performance of Convolutional Neural Networks via Distillation"
excerpt: "[ICCV2019]Be Your Own Teacher: Imporve the Performance of Convolutional Neural Networks via Distillation"
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

한빈님이랑 ICCV 준비 하면서 메인으로 영감을 받는 논문. 내용은 쉽고 구현도 깔끔하게 되어있어서 보기 편하다. 그런데 성능이 좀 높게 나온것 같다. 베이스라인 자체가 높아서 어떻게
구현 했는지를 봐야됨 (Augmentation 이라던가..). 

### Abstract & Introduction
KD 영감 받아서 Teacher Student 구조를 쓰는 것이 아니라, self-distillation 프레임 워크를 새로 만들었다. 당연히 성능향상이 있고, 
Teacher를 따로 훈련하지 않는다는 점에서 좋다. 지금까지 CNN이 발전하면서 성능을 늘리는 것은 대부분 모델의 구조보다는 크기에 관련이 되는데
 크기가 커질수록 성능은 좋아지지만 비용에 비해서는 비효율적이다.

### Self distillation
사실 이 논문을 읽는데는 3시간도 안걸렸다. 왜냐하면 이해하기 어려운 수식도 없고 그냥 모델 구조 그림에서 모든것이 설명 가능하다. 그만큼 단순하지만 성능 
향상이 뛰어나고 아이디어 또한 매우 경제적이다. 그래서 자세한 내용은 본문을 보고 그림한에서만 설명

![](/assets/images/2021-08-14-Be_Your_Own_Teacher/1.PNG)
위의 그림은 resnet18을 기반으로 한 것이다. resnet18은 4개의 resblock으로 되어있고 각 resblock에서 생성된 4개의 feature를 이용한다. 우리가 흔히 사용하는 (
deepest classifier라고 표현 )마지막 feature는 마지막 classifier fc로 들어가고, 새롭게 소개한 3개는 shallow classifier라는 새로운 bottlenet 구조에
 들어간 뒤 fc로 들어가서 logit을 낸다. 중요한 점은 bottle낵에서 나온 feature를 이용해야 하기 때문에 각각의 resblock에서 나온 feature의 채널을 맞추기 위해
  1x1 conv를 사용하였다. (정확히는 512를 사용하였으므로 512x1x1의 모양이다). 또한 shallow classifier1 (맨앞의 bottleneck1 부분)은 더 깊은 bottleneck구조를
   사용하였고, 뒤로 갈 수 록 더 단순한 bottleneck을 사용한다. 사용한 총 loss는 10개. deepest classifier와 모든 shallow classifier를 비교한다. loss는 크게 
   3개 => CE, KD, Feature loss.

CE는 맨 아래의 softmax를 거친 shallow classifier 3개의 deepest classifier1개를 원래의 원핫 인코딩된 label과 계산하고, KD는 deepest classifier의 softmax와
 각 shallow classifeir의 logit을 결합하여 KL divergence loss 3개, 마지막으로 resblock4에서 나온 feature와 bottleneck을 통과한 feature를 비교하여 3개가 나온다.
  Feature loss는 L2 loss를 사용하였다. 


### Experiments
가장 중요한 결과는 성능이 올랐다는 것. 이 결과를 뒷받침 하는 것으로 Scalable Depth를 설명한다. 아래 표를 보면 classifier1/4가 제일 빠른데 비록 shallow classifier1이 가장 복잡한
 convolution이라도 resblock을 1개만 거치므로 아래와 같이 가장 빠르다고 평가할 수 있다. 다만 성능은 가장 안좋다. 
![](/assets/images/2021-08-14-Be_Your_Own_Teacher/2.PNG)


아래 그림처럼 저자들이 주장하는 large model과 shallow model의 성능차이가 minimum을 형성하는 그래프의 모양이 달라서 그렇다는 것도 주목할만하다. 
![](/assets/images/2021-08-14-Be_Your_Own_Teacher/3.PNG)