---
title:  "[NIPS2021]Denoised Smoothing: A Provable Defense for Pretrained Classifiers"
excerpt: "[NIPS2021]Denoised Smoothing: A Provable Defense for Pretrained Classifiers"
categories:
  - Paper & Knowledge
  
tags:
  - Denoise
 
published: false
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-18T23:35:00-05:00
---

### Abstract & Introduction
딥러닝 분류기는 발전해왔는데 adversarial attack이 들어가면 성능저하 심하다는 점. 이 논문의 시나리오는 배포된 분류기를 robust하게 만들고 
싶은데 retrain하기에는 비용도 많이 들고, 이러한 수동적인 학습은 다른 공격에 취약하기 때문에 generalization이 떨어진다는 것이 문제이다.
 그래서 이 논문에서는 2가지 시나리오를 제시. 첫번째는 기존의 모델에 접근할 수 있는 white box, 두번째는 주어진 모델이분류만 할 수 있도록 
 모델 내부에는 관여하지 않는 black box를 제시. 물론 white box가 더 나은 성능을 낼 수 있다. 또한 모델의 재 훈련은 최소화하기를 원함. 
 
### Denoised Smoothing
#### Background for Randomized Smoothing
먼저 이 방법론을 이해하려면 배경지식으로 Randomized Smoothing을 먼저 보자. 일단은 구글링해도 자세히 나오는 설명이 없어 이 논문에서만의 
설명만 쓴다.\
$f: given\;classifier, g: robust\;classifier$ 일때,
<center>$g(x) = argmax_{c \in y} P[f(x+\delta) = c] where \delta \sim \mathcal{N}(0, \sigma^2 I)$</center>
여기서 $\sigma$는 robustness와 accuracy의 tradeoff를 조절한다. 즉, 증가하면 robustness가 증가한다. (노이즈가 많아 지면 모델이 더 튼튼해진다고 생각. 대신 정확도는 낮아짐.)\
중간에 얼마나 robust한지를 나타내는 측도가 나온다. 가장 높은 확률을 나타내는 $p_A$와 두번째로 높은 확률을 나타내는 $p_B$에 따라서 다음과 같다.
<center>$R = \frac{\sigma}{2}(\Phi^{-1}()p_A) - \Phi^{-1}(p_B)$$.</center>
내가 이해하기로는 $N(0, \sigma^2)$는 반지름이 $\sigma$인 원인 노이즈들의 집합이 그려지는데 이 노이즈가 더해지고, $p_A - p_B$만큼의 robust가 생긴다고 생각이 든다. $\sigma$가 커질수록 robust가 커지므로 비례함.
* 여기서 Monte Carlo Sampling을 곁들여 $p_A$와 $p_B$를 설명하는데 Monte Carlp sampling은 우리가 흔히 아는 무작위 샘플링, 즉 어떤 확률 공간안에서 막 뽑다 보면 대충 윤곽이 그려진다. 이 샘플링으로 계속 뽑다 보면 
lower bound, upper bound를 추정 할 수 있다. 지금은 가정을 하는 것이기 때문에 최악의 상황을 가정한다. 즉, 가장 높은 확률과 두번째의 확률의 거리가 가장 작을 때를 가정하므로 
$p_A$의 lower bound $\bar{p_A}$, $p_B$dml upper bound $\underline{p_B}$를 가정한 것이다.

#### Image Denoising: a Key Preprocessing Step for Denoised Smoothing
위에서의 smoothing방법의 단점은 분류기가 미리 gaussian noise augmentation을 적용해야 한다는 것이다. 이 논문에서 제시한 방법론은 이러한 가정이 없어도 
된다는 것을 보여준다. 먼저, 주어진 이미지에서 gaussian noise를 제거한다. 다시 말하면, 위의 method는 gaussian noise를 붙인 이미지가 들어오는 것을 예측하고
훈련시키고, denoising smoothing은 아예 이런 경우를 없애는 것이다. 따라서 $D_{\theta}: denoiser, f: classifier$라 할 때, 
<center>$g(x) = argmax_{c \in y}P[f(D_{\theta}(x+\delta)) = c]\;where\;\delta \sim \mathcal{N}(0, \sigma^2 I)$</center>
위의 식을 보면, robust분류기 g는 먼저 noise를 제거하고 이것을 다시 기존의 분류기에 넣은 결과의 argmax를 뽑는 분류기이다. 즉, 바로 전에 말한 노이즈를 모두 제거한 것을 입력으로 하는 것과 동일하다.
 
#### Training the Denoiser $D_{\theta}$
denoiser $D_{\theta}$를 훈련 시키기 위해 loss function 2개를 소개한다.
 
##### MSE objective
이 함수는 노이즈가 있는 입력을 디노이즈 하는 것이다. 즉, 디노이징의 가장 필수적인 목표를 목적으로 하는 함수.
<center>$L_{MSE} = E_{S, \delta}\Vert D_{\theta}(x_i + \delta) - x_i \Vert_2^2$$</center>

##### Stability objective
이 함수는 기존의 분류기의 성능을 떨어뜨리지 않게 하기위함이다. 기존의 분류기가 noise augmentation을 적용했을 때, denoise를 거치면 노이즈가 없는 이미지가 되므로 기존 분류기와는 조금 다른
성질이 되어버린다. 따라서 기존 분류기의 prediction을 따라가는 손실 함수를 만든다.
<center>L_{stab} = E_{S, \delta}L_{CE} (F(D_{\theta} (x_i + \delta), f(x_i))\; where\; \delta \sim \mathcal{N}(0, \sigma^2 I)</center>
$F(x)$는 기존의 분류기가 argmax를 거치지 않은 값, 즉 확률 값이고, $f(x)$는 원 핫 레이블이다. 

introduction에서 white box, black box두 가지의 시나리오를 가정 한다 했는데, white box일 때는 두 손실함수로 denoiser와 classifier를 둘 다 훈련 시키고, black box에서는 denoiser만 훈련시킨다.
즉, white box일 때에만 f의 역전파를 열어둔다. 