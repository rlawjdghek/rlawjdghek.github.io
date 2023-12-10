---
title: "[NeurIPS2017]Toward Multimodal Image-to-Image Translation"
excerpt: "[NeurIPS2017]Toward Multimodal Image-to-Image Translation"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-06-16T15:04:00-05:00
---

BycicyleGAN은 paired I2I translation이기 때문에 큰 한계점이 존재 하지만, multimodal에 대한 좋은 분석과 흐름을 갖고 있어 공부용으로는 의미있게 읽었다. 
MUNIT과 DRIT++을 먼저 읽어서 이미 익숙한 그림과 학습 방법이 있었으나 알고 있던 방법들이 잘 정리되는데에 큰 도움이 되었다. Introduction이나 실험은 제외하고 메소드 부분에서 
시사하는 바를 메모하였다. 또한 이 논문에서 새롭게 배운것은 random distribution $N(0, 1)$과 어떤 스타일 distribution $P$가 KL로 맞춰질 때, 손실 함수를 어떻게 구현하는지를 증명해 두었다.

![](/assets/images/2022-06-16-BicycleGAN/2.jpg)
# Multimodal Image-to-Image Translation
Pix2Pix는 가장 기본적으로 이미지 하나에 대하여 정답을 요구하고 모델 훈련시에도 출력을 정답에 맞추도록 학습된다. 실세계에 비추어 보았을 때, 이것의
근본적인 문제점은, 구두의 스케치가 들어왔다고 해보자. 스케치만으로 나올수 있는 실제 컬리 이미지의 종류는 무한히 많이 존재한다. 하지만 pix2pix는 그런것을 무시하고 어떤 정답 이미지만으로
모델이 구두의 스케치를 정답에 맞추려고 한다는 것이다. 이는 결국 mode collapse가 더 쉽게 발생하도록 하고, unimodal, dual-domain이라는 가장 제한적인 가정을 만들어낸다. 
따라서 등장한 것이 Multimodal, 즉, 어떤 도메인에서 다른 도메인으로 갈 때, 나올 수 있는 경우의 수를 여러개로 가정한다. 이 떄 반드시 필요한 것은 우리가 정답을 여러개 만드는 것은 결국
나올 수 있는 이미지의 경우의 수에 제한을 거는 것이므로 우리가 조절 할 수 없는 새로운 입력 또는 출력을 활용해야 한다. 이 때 도입된 것이 random noise vector이다. 이 벡터는 대부분 
표준 정규 분포를 따른다. 이 random vector를 어떻게 활용하여 multimodal을 도입하느냐가 모델 설계의 관건인데, BiCycleGAN에서는 처음으로 MUNIT, DRIT++등에서 볼 수 있는 테크닉들을 
대중적으로 소개하였다. 실제로는 기존에 있는 개념을 잘 정리했다고 생각됨. 

이제 그림의 (a)와 (b)그림을 보자. pix2pix에서 가장 직관적으로 추가적인 입력으로 noise vector를 추가했다고 하면(더하거나, concatenation), 이것도 multimodal의 개념을 사용했다고 할 수 있다.
하지만 이 접근의 문제는 multimodal conditional GAN의 하나인 content이미지의 강력한 prior이다. 모델이 학습하기 전에 입력과 noise vector로부터 출력을 학습해야 한다. 하지만 모델의 입장에서는
정보가 없는 noise vector보다 정보가 많은 입력 (content)이미지를 활용하여 출력을 만들어 낼 것이다. 이 때 입력 이미지에 대한 의존도가 높아지고, 결과적으로는 noise vector는 단지 denoising을 
하기 위한 수단이 될 뿐, multimodal로써의 도구로는 힘을 잃는다. 따라서 본 논문에서 활용한 방법론은 아래와 같다.

BicycleGAN은 기존의 2가지 모델, cVAE-GAN, cLR-GAN을 같이 사용하여 성능을 더하였다. 
### Conditional Variational Autoencoder GAN (cVAE-GAN)
위 그림의 (c)이미지를 보자. VAE와 마찬가지로 입력 이미지로부터 정규분포의 평균($\mu$)와 표준 편차($\sigma$)를 출력한다. 구현상 빠르게 하기 위해서 정확하게는 평균과 분산에 로그를 취한 값(logvar=$\log(\sigma^2)$)을 
출력한다. 표준 정규분포 $N(0,1)$을 따르는 z를 생성하고, 출력된 2개의 모수를 활용하여 reparametrize한다. 그러면 완성된 vector는 $N(\mu, \sigma^2)$를 따르게 된다. 
이제 decoder가 테스트 때 스타일벡터와 realA를 사용하여 다양한 출력을 해야한다. 이 때 스타일 벡터는 reference image에서 인코더를 통과해서 나온 벡터가 될 수도 있지만, reference image의 스타일 벡터와
표준 정규분포를 맞춰줌으로써 표준 정규분포가 스타일 벡터의 집합이 된다. 따라서 추론 때에는 reference image 또는 표준 정규분포 2가지의 경우를 모두 사용할 수 있다.  
여기서 한가지 주목할 점은 훈련때 사용한 reference image에서부터의 스타일벡터와 표준 정규분포를 KL loss를 활용하여 맞춰준다. 구현 상으로는 
```python
loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu**2 - logvar + 1)
```
인데, 왜 그렇게 되는지 증명하면 아래와 같다.
![](/assets/images/2022-06-16-BicycleGAN/1.jpg)

첫번째 이미지 (c)그림에서 볼 수 있듯이, 전체적인 흐름은 $realB \rightarrow z \rightarrow geneB$이다. 

### Conditional Latent Regressor (cLR-GAN)
첫번쨰 이미지 (d)그림에서 보면, $z \rightarrow geneB \rightarrow genez$이다. 이름과 걸맞게 latent z를 회귀하여 genez와 L1으로 맞춰준다. 

### BicycleGAN
cVAR-GAN, cLR-GAN을 조합하여 BicycleGAN을 만들수 있다. BicycleGAN의 구조를 스케치하면 아래와 같다. 
![](/assets/images/2022-06-16-BicycleGAN/3.jpg)
위에는 cVAE-GAN, 아래는 cLR-GAN이라 생각하면 된다. 즉, 공통된 Encoder와 Decoder를 사용하여 구현되었다. 이는 MUNIT, DRIT++과 아주 비슷한 흐름인 것을 알 수 있다.
하지만 BicycleGAN은 cVAE-GAN을 사용했기 때문에 cVAE-GAN 부분에서 realB와 geneB가 L1 loss임을 볼 수 있다. 즉, realA와 styleB를 합친것이 정답인 realB와 같아야하고, 이는 realA의 content가 
realB의 content와 같다는 것을 의미한다. 즉, realA와 realB는 content가 일치하는 paired dataset이다. 

# Experiment
본 논문에서는 Multimodal을 거의 처음 제시했기 때문에 베이스라인이 pix2pix+noise, cVAE-GAN, cLR-GAN이다. 성능은 BicycleGAN이 제일 좋음
![](/assets/images/2022-06-16-BicycleGAN/5.jpg)

또 한가지 주목해야 할 점은, latent z의 길이이다. z의 길이가 길수록, 차원이 높아진다는 것이고, multimodal로 줄 수 있는 다양성이 높아진다는 것을 의미한다.
본 논문에서는 2, 8, 256으로 실험을 했고, 아래 그림과 같이 너무 낮은 길이 2에서는 벽돌의 질감같은것이 굉장히 단순하다.
반면, 높은 차원 256에서는 너무 복잡하여 오히려 현실성이 없는 이미지를 생성하였다.
![](/assets/images/2022-06-16-BicycleGAN/4.jpg)
