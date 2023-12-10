---
title: "[ICLR2022]InfinityGAN : Towards Infinite-Pixel Image Synthesis"
excerpt: "[ICLR2022]InfinityGAN : Towards Infinite-Pixel Image Synthesis"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-05-22T15:04:00-05:00
---

ICLR2022 google에서 저술한 논문 찾다가 발견한 논문. 이제 고정된 사이즈로 이미지를 생성하는 것은 StyleGAN2로써 대부분 끝난 것 같고, Image inpainting이나, outpainting 등이 활성화 되는듯 하다.
이 논문에서는 이미지 outpainting을 넘어서 초고해상도를 생성하기 위해 무한대로 확장시킬 수 있는 모델을 제시한다. StyleGAN2는 고정된 1024x1024를 생성하는 반면, InfinityGAN은 작은 패치를 여러개 생성해서
이어붙인다. 따라서 논문에서는 12GB의 GPU로도 학습과 추론을 할 수 있다고 함. 

# Abstract & Introduction
이 논문에서 해결한 문제들은 아래와 같다. 
1. 기존의 고해상도 이미지를 생성하는 모델은 많은 자원을 소모하고, 특정이미지 스케일만 생성할 수 있다는 한계점이 존재한다. 하지만 패치단위로 추론한다면 시간은 오래 걸리지만 큰 VRAM을 필요로 하지 않는다.
2. 큰 이미지는 global과 local consistency가 갖춰져야 한다. 이로 인하여 더 현질적이고, 반복적인 패턴을 피할 수 있다. 
3. InfinityGAN은 다양한 application에 적용가능하며, 각 application에서 좋은 이례없는 성능을 보인다. 

일반적으로 모델이 무한의 크기를 생성하기 위해서는 무한의 입력을 받아야 한다. 하지만 이는 불가능하므로 모델이 추론시에 직접적인 supervision없이 implicit global sturcture를 배워야한다. 즉, 여기서는
추론할 때에 입력으로 위치 좌표를 condition으로 주는데, 이렇게 작은 연관성이라도 있는 사전정보를 주고, 그 사전정보로부터 global, local하게 일관성 있는 이미지를 생성할 수 있어야 한다. 

InfinityGAN의 아키텍쳐는 크게 2가지로 나누어 진다. 
1. structure synthesizer : 이미지를 생성하기 전 대부분의 사전정보를 구축한다. 본 논문에서 정의하는 사전 정보로는 global style (이미지를 구성하는 전체적인 스타일. 예를 들어, 중세시대 그림스타일 또는 해안가),
local structure (지역적인 부분에서 물체의 shape)으로 나눌 수 있다. 이 아키텍쳐는 전체적인 스타일을 결정하는 latent와 local에 대한 latent, 위치 좌표를 입력으로 받는다. 추후 모델파트에서 더 자세히 다룬다.
2. texture synthesizer : 1번 아키텍쳐에서 전체적인 구상을 latent화 했다면, 이 아키텍쳐에서는 이를 입력으로 받아 다양한 texture를 고려한 뒤 실제 이미지를 생성한다. 


![](/assets/images/2022-05-23-InfinityGAN/1.JPG)
# Method 
위에서 언급하였듯이, 이미지는 global정보와 local정보를 갖는다. <br/>
먼저 global 정보는 본 논문에서 자주 사용하는, holistic appearance가 이미지 전반적으로 일관성 있어야 한다. global 정보는 무한의 크기 이미지 모두 일치해야 하기 때문에
추론 과정에서 항상 같은 latent로 입력된다. <br/>
다음으로 local정보로는, 이미지를 지역적으로 볼 때 담겨있는 물체의 structure와 texture를 지칭할 수 있다. structure는 알다시피 물체의 shape이나 arrangement (배열)등을 나타내고, texture는 우선적으로 structure가
잡혀야 그 다음으로 생각할 수 있다. texture또한 하나의 물체에 대하여 일관성이 있어야 한다. 먼저 물체의 shape이 잡히면, 그 물체에 texture를 입혀야 하는데, texture는 structure뿐만 아니라 global 정보인 holistic appearance
와도 어느정도 일치해야 한다. 결론적으로 holistic appearance, structure, texture는 모두 상관관계가 있고, 모델이 여러 패치를 생성한다 할지라도 앞의 3가지 정보를 모두 일관성있게 반영해야한다. 

### Structure Synthesis
반복해서 말하지만, 본 논문에서 고려하는 이미지의 3가지 요소는 global style, local structure, local texture이다. Structure Synthesis모듈에서는 global style과 local structure를 구성하는 latent를 만든다. 
식을 먼저 적자면 아래와 같다.
\begin{equation}
z_S = G_S(z_g, z_l, c)
\end{equation}
$z_g$는 global style을 담당하는 latent vector, $z_l$은 local structure를 담당하는 latent tesnor, $c$는 위지좌표이다. 따라서 $z_S$는 이 모든것이 합쳐진 structural latent variable이라 할 수 있다. 또한 이것이 
다음 네트워크 texture synthesis의 입력중 하나가 된다. <br/>
주목할 점은 $z_g$는 흔히 사용하는 벡터인데, $z_l \in \mathbb{R}^{H \times W \times D_{z_l}}$인 3차원 텐서이다. 그림을 보면 $z_l$은 생성 이미지의 좌표 $c$를 받고, 이는
이미지가 생성된 이후 패치와도 동일한 위치에 있는 것을 볼 수 있다. <br/>
다음으로 $c$는 target patch의 위치를 나타낸다. 우리가 훈련 데이터로 사용하는 이미지는 해상도가 높지만 실제 모델 입력에는 패치화 되서 들어가기 때문에 큰 원본이미지의 원점을 중심으로 패치의 위치좌표를 구할 수 있다. 
위치 좌표는 사전정보로 이루어져 있다. <br/>
\begin{equation}
c = (tanh(i_y), cos(i_x / T), sin(i_x/  T))
\end{equation}
본 논문에서는 데이터셋을 자연경관, 건축물 등만 사용하였다. 데이터셋의 다양성 부족을 의미하기 때문에 한계점이라 볼 수 있지만, 논문에서 제시하는 사전정보와 잘 맞기 때문에 성능은 증가될 수 있다.
우리가 흔히 자연경관을 볼 때 (또는 Flicker-landscape 이미지를 볼 때) 윗부분은 하늘, 아랫부분은 물 또는 땅으로 볼 수 있다. 가장 diversity가 큰 부분은 중간 부분이다. 본 논문에서 보여주는 Flicker-landscape 데이터셋은
이미지가 대부분 이런 구성으로 되어있다. 따라서 위의식 $c$를 볼 때, y좌표는 원점에서 멀어질수록 (i.e., 하늘 또는 물, 땅으로 갈수록) 일정한 값을 갖는 것을 알 수 있다. x축은 sinusoidal로 구성되어 있는데, 이것또한 
자연경관의 경우 대부분 수평에는 비슷한 경관이 있다는 사전정보를 활용한 듯 하다.

패치 단위로 추론을 하는 경우 발생하는 고질적인 문제중 하나는 반복되는 패턴 생성이다. 따라서 본 논문에서는 생성 이미지들간의 diversity를 증가시켜주는 mode-seeking diversity loss를 사용하였다. 식은 아래와 같다. 
\begin{equation}
\mathcal{L} = \parallel z\_{l_1} - z\_{l_2} \parallel \_1 / \parallel G\_S(z\_g, z\_{l_1}, c) - G\_s(z\_g, z\_{l_2}, c) \parallel
\end{equation}

또한 StyleGAN2와 같은 모델은 입력 latent z가(input query) 독립이다. InfinityGAN에서는 $z_l$에 해당하는데, 이렇게 입력이 독립적으로 들어간다면, 이미지를 확장할 때, 불안정할 수 있다. 
따라서 본 논문에서는 feature unfolding technique를 사용한다. Feature unfolding technique는 $z_l$과 $c$에 대하여 인접한 정보를 더 잘 설명하게 하고, 이는 모델이 더 큰 receptive field를 갖게 하는것과 마찬가지의
효과를 준다. 구현에서는 STructure Synthesis모델의 각 레이어에 $k \times k$의 feature unfolding transformation을 각 위치에 준다. 얻어진 unfolded input $f'$식은 아래와 같다.
\begin{equation}
f'\_{i,j} = Concat({f(i+n, j+m)})\_{n,m \in {-k/2, k/2}}
\end{equation}

### Texture Synthesis
Structure Synthesizer는 이미지의 전체적인 틀을 잡는다. 이제 이 틀로 Texture를 부여하는 Texture Synthesizer를 학습한다. <br/>
Texture Synthesizer는 StyleGAN2모델 구조를 거의 그대로 가져다 사용한다. 단 성능에 큰 영향을 미치는 중요한 특징이 있으므로 스킵하면 안된다. 먼저 texture synthesizer의 forward한 패치 ($p_c$)를 단순식으로 나타내면 아래와 같다.
\begin{equation}
p_c = G_T(z_S, z_g, z_n)
\end{equation}
$z_S$는 structure synthesizer가 만든 전체적인 틀 latent이고, $z_g$는 global style, $z_n$은 styleGAN2에서 제시한 fine-grained texture를 위한 noise이다. <br/>
StyleGAN2로부터의 변경사항을 나열하자면 아래와 같다.
1. Fixed constant input을 $z_S$로 바꾸었다. 전체적인 틀이 $z_S$이므로 이것을 베이스로 깔고 들어간다고 생각하면 된다. 
2. $z_g$가 mapping network의 입력으로 들어간다. 이미지 전반적인 스타일을 일관적으로 유지하기 위해서라 생각하면 된다. 
3. <u>zero-padding을 삭제하였다.</u>

3가지중 1번, 2번은 단순 입력만 대체하였기 때문에 설명할 것이 없고, 가장 중요한 것은 3번, zero-padding을 없앴다는 것이다. <br/>
일반적으로 CNN에서 zero-padding의 역할은
1. edge에서의 convolution kernel 영향력이 균형있게 들어갈 수 있다.
2. 위치정보를 기억하게할 수 있다.
3. 이미지의 크기를 유지할 수 있다. 
4. 이미지 feature map에 유동성없이 일정하게 반영된다. 
로 정리 할 수 있는데, 3번은 크게 상관없고 1번또한 중요하게 고려할 사항이 아니다. 우리가 주목해야 할것은 2번과 4번이다. <br/>
먼저 2번은 앞서 structure synthesis모델로부터 $c$가 담당하고 있기 때문에 zero padding이 주는 상대적인 위치는 아주 치명적이라 할 수 있다. zero padding이 주는 edge의 위치정보는 어느 이미지가 들어오던간에
항상 일정한 위치에 생성되므로 상대적인 위치와 다양한 크기의 이미지를 만드는 현 task에서는 성능 하락의 주 원인이 된다. <br/>
다음으로 4번 또한 유동성 있는 패치단위의 이미지 생성에서 악영향을 줄 수 있다. 위에서 언급하였듯이, outpainting의 고질적인 문제중 하나는 반복적인 패턴의 생성이다. 패치단위로 생성하면서 서로 일관성은 있지만 
같은 이미지를 생성해서는 안된다. zero-padding은 항상 0값을 edge에 부여함으로써 일관된 패턴을 주입한다. 따라서 본 논문에서는 임의의 이미지 사이즈에 대한 다양한 패치의 생성을 위해서 zero-padding을 삭제하였다. 

### Model Training 
D는 StyleGAN2와 비슷한 것을 사용. non-satuating loss $\mathcal{L}\_{adv}$, R1 regularization $\mathcal{L}\_{R}$, path length regularization $\mathcal{L}\_{path}$를 사용. 또한 위치정보를 부여하는 과정에서 사용한
사전정보, y축으로 satuation (tanh)함수를 사용한 것을 다시 이용하여 auxiliary task로 사용한다. 즉, y축 위치를 regression하는 task를 더 해결한다.
\begin{equation}
\mathcal{L}\_{ar} = \parallel \hat{c}\_y - \tilde{c}\_y \parallel _1
\end{equation}
$\hat{c}_y$는 auxiliary D에 의하여 예측된 y좌표이고, $\tilde{c}_y$는 실제 tanh값 또는 full image의 패치의 y좌표이다. 종합적으로 전체 손실 함수는 아래와 같다.
\begin{equation}
\min\_D  \mathcal{L}\_{adv} + \lambda\_{ar}\mathcal{L}\_{ar} + \lambda\_{R_1}\mathcal{L}\_{R_1}
\end{equation}
\begin{equation}
\min\_G = -\mathcal{L}\_{adv} + \lambda\_{ar}\mathcal{L}\_{ar} + \lambda\_{div}\mathcal{L}\_{div} + \lambda\_{path}\mathcal{L}\_{path}
\end{equation}

