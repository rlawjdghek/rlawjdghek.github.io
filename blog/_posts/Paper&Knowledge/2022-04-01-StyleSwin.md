---
title: "[CVPR2022]StyleSwin: Transformer-based GAN for High-resolution Image Generation"
excerpt: "[CVPR2022]StyleSwin: Transformer-based GAN for High-resolution Image Generation"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-04-01T15:04:00-05:00
---
![](/assets/images/2022-04-01-StyleSwin/8.PNG)
![](/assets/images/2022-04-01-StyleSwin/3.PNG)
![](/assets/images/2022-04-01-StyleSwin/4.PNG)
내 구현에서 사용한 모듈들. StyleSwinBlock에서 double attention을 구현하기 위해 WindowAttention의 입력이 qkv가 들어간다. (swin의 WindowAttention은 x가 입력.)
TransGAN은 1024이미지의 생성을 하지 않았다. 직접 구현해본 바로는 swin transformer를 활용해도 1024이미지까지에서는 불가능한듯. 이 논문은 StyleGAN2를 swin transformer로 완성한 논문이다. TransGAN은 swin transformer
이후 처음 GAN을 트랜스포머를 활용하여 만들었다면, 이 논문은 트랜스포머를 활용한 GAN이 CNN보다 더 좋아질 수 있다는 것을 보여준다. 중간에 double attention이 나오면서 구현상에 swin transformer와 다른점이 있다. 자세히 말하면,
swin transformer에서는 window partition을 먼저 진행한 다음에 qkv레이어를 통과하여 qkv를 만들지만, 여기서는 먼저 채널을 2로 나누어야 하기 떄문에 qkv레이어를 통과한 뒤, 각각의 qkv를 window partition한다. 하지만 둘 다 결과는 
같다. 단지 double attention을 강조하기 위하여 이렇게 구현한듯. 또한 TransGAN에서는 pixelshuffle을 활용하여 채널을 줄였지만, 여기서는 StyleSwinBlock 다음의 bilinear에 linear로 2씩 줄인다. 
구현은 [링크](https://github.com/rlawjdghek/Generative_Models/tree/main/GANs/StyleSwin) 참조.

### Abstract & Introduction
local attention이 트랜스포머 기반 생성모델에서는 필수적이다. 하지만 이는 비용이 상당하기 때문에 현재 있는 자원으로는 조절하여 사용해야 한다고 한다. local attention은 CNN에서와 같이 local inductive bias를 
야기하는데 이는 GAN 학습에 더 좋을 수도 있으나, 너무 많은 bias가 있을 경우 오히려 생성 이미지의 저하를 야기한다. 따라서 이 논문에서는 swin transformer에서 사용한 shift를 활용한다. swin transformer에서 제시한
블록의 구조를 그대로 사용하면, 먼저 window multi-head self attention (W-MSA)를 통과한 뒤 shift W-MSA를 통과하여 연산량이 배로 늘어난다. 1024x1024의 고화질의 이미지에서는 이러한 구조에 단점이 확연히
드러나기 때문에 여기서는 각 블록에서 먼저 채널을 2로 나누고 W-MSA와 SW-MSA를 병렬로 사용한다. 또한 swin transformer에서 사용한 relative position encoding을 그대로 사용하지 않고 zero padding을 대체할 absolute position 
encoding을 사용한다. <br/>
논문에서 주장 및 제시한 contribution을 나열하면 아래와 같다.
1. local attention : G의 capacity를 증가시키지만, receptive field를 제한한다.
2. double attention : local attention만 사용할 경우 발생하는 단점 (제한된 receptive field)에 대하여 낮은 비용으로 해결한다. 
3. sinusoidal  positional encoding : CNN은 위치를 추정하기 위해 zero-padding을 사용한다. 기존의 트랜스포머는 이런것이 없으므로 sinusoid를 제안한다.
4. wavelet discriminator : 고해상도(512 이상) 이미지에서 발생하는 blocking artifact를 해결한다. 

### Method 
![](/assets/images/2022-04-01-StyleSwin/1.PNG)
#### Transformer-based GAN architecture
먼저 모든 레이어는 swin transformer의 모듈을 사용한다고 보면 된다. 이 논문은 swin transformer와 stylegan2의 깃헙을 짬뽕해놓은 느낌. 위의 그림에서 (b)를 보면 stylegan2와 같이 mapping network를 그대로 사용하고
중간에 adaptive instance normalization (AdaIN)을 사용한 것을 볼 수 있다. <br/>
또한 D는 트랜스포머의 장점을 살리기 위해 기존 CNN구조를 사용하였다. 고화질 이미지에 대해서만 Wavelet을 사용하였다. 저자들은 D 또한 트랜스포머로 구현하는 것이 성능이 더 좋다고 말한다. 하지만 이 논문에서는 완성도가
G에 적용된 트랜스포머에 의하여 향상된다는 것을 보여주기 위해서 기존의 CNN을 사용하였다. 

![](/assets/images/2022-04-01-StyleSwin/5.PNG)
#### Style Injection
Style Injection을 하기 위해서 마찬가지로 AdaIN을 사용하였다. 여러 정규화 기법들로 ablation 실험을 진행하였으나 AdaIN이 가장 좋은 성능을 보였다. 중간의 AdaBN은 IN대신 BN을 사용한 것인데, 고화질 GAN 특성상 배치 
사이즈를 크게 사용할 수 없어 효과가 없다. 

#### Double Attention
메소드 첫부분의 그림에서 (c) 그림과 맨 위의 모델의 자세한 구조에서 StyleSwinBlock을 참고하자. swin transformer와 달리 먼저 qkv layer를 통과시킨뒤 채널을 둘로 나누어 두개의 qkv로 만든다. 그 다음 attn레이어를 
지나기 전에 q,k,v를 각각 window화 해준다. 하나는 일반적인 attn을 지나고 나머지는 shift되어 들어간다. 마지막에는 다시 concatenate해서 채널을 맞춰줌. 논문에서는 식 (3)과 같이 헤드에 3개의 W가 들어가는데
사실상 연산은 qkv layer먼저 하나 window partition화를 먼저하니 값이 똑같이 된다. shape는 [BS*nW x win**2 x C]로 같다.

#### Local-global Position Encoding
Relative Position encoding은 transformer의 성능을 대폭 상승시켰다. 하지만 ConvNet에서 컨볼루션 연산에서 사이즈를 유지하고 이미지의 끝이라고 알려주는(항상 이미지의 끝에서 zero-padding이 발생하므로 
절대적인 위치라고 할 수 있다.) zero-padding의 역할을 부여하기 위해서 저자들은 sinusoidal position encoding을 각 스케일에서 제안한다. 이는 구현상에서 BilinearUpsample 모듈에 들어있는 것을 확인할 수 있다.

![](/assets/images/2022-04-01-StyleSwin/6.PNG)
### Blocking artiface in high-resolution synthesis
위와 같은 구조로 G를 만들었다면 고화질의 이미지에서 패치화로 인한 blocking artifact(이미지에서 체크무늬형상)를 볼 수 있다. 위의 그림의 아래 확대된 이미지들을 보자. 이는 1차원에서도 확인 할 수 있는데, 아래 그림을 
보면 직관적으로 이해할 수 있다.

![](/assets/images/2022-04-01-StyleSwin/7.PNG)

그림에서 볼 수 있듯이, local-attention은 명확하게 local한 부분끼리의 일관성을 파괴한다. 즉, 들어온 local 입력만에 집중하여 입력간의 관계를 무시한다. 2D에서는 JPEG compression 효과가 나타난다. 이를 방지하기 위해 저자들은 
1. Patch-Discriminator
2. Total variation annealing
3. Wavelet Discriminator
를 시도하였다. 1번은 효과가 약간, 2,3번은 효과가 좋았으나 2번은 생성 이미지의 퀼리티에 영향을 주어 최종적으로는 3번을 채택하였다. 

![](/assets/images/2022-04-01-StyleSwin/2.PNG)
### Experiments
성능은 FFHQ, LSUN, CelebA-HQ에서 좋은 성능을 내었고, 한가지 주목할 점은 연산량이다. 위의 표에서 볼 수 있듯이, StyleGAN2와 비슷한 연산량을 자랑한다. 즉, 기존 CNN과 비슷한 규모의 네트워크를 만들 수 있다는 사실을 보여준다.




 


