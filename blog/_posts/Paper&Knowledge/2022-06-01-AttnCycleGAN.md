---
title: "[NeurIPS2018]Unsupervised Attention-guided Image-to-Image Translation"
excerpt: "[NeurIPS2018]Unsupervised Attention-guided Image-to-Image Translation"
categories:
  - Paper & Knowledge
  
tags:
  - GAN
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-06-01T15:04:00-05:00
---

CycleGAN에서 attention module만 추가하여 발전시켰다. 단, CycleGAN과 같이 특정 object의 형체는 지키면서 texture만 바꾸는 task에 적합할 듯 하다. 
CycleGAN의 발전형이므로 크게 어려운 것은 없다. 코드는 [링크](https://github.com/rlawjdghek/Generative_Models/tree/main/GANs/Attention_CycleGAN) 참조.

# Abstract & Introduction
CycleGAN은 generation과 reconstruction에서 두개의 generator가 최대한 reconstruction을 잘 시키기 위해 형체는 유지한다. 하지만 직관적으로는 이렇게 생각할 수 있지만, 좀 더 구체적으로 조절 할 수 있다면
좋을 듯하다. PatchGAN은 생성된 이미지를 지역적인 부분의 패치만 보고 discriminator가 판별하는데, 이 방법론 또한 명확히 물체에 집중하라는 학습 방법을 제공하지는 않는다. 
반면, 이 논문에서는 attention을 활용하여 모델이 학습하는 동안 집중하는 부분은 변형시키고 (다른 말로, generation, reconstruction에 주로 기여하는 부분에 집중한다.) 나머지는 바꾸지 않도록 조절하는 방법을 제시한다. 
이렇게 특정 부분에 대하여 집중적으로 변형시키는 방법으로, 학습 도중 중요하지 않은 부분을 Generator가 변형시키는 것을 완화할 수 있고, 더욱이 변형된 부분에 대하여 원본의 배경을 마스크 연산으로 덮음으로써 객체 이외의 부분을 
보존할 수 있다.

![](/assets/images/2022-06-01-AttnCycleGAN/1.jpg)
# Method 
### Attention-guided Generator
그림을 보면 전제적인 학습은 기존 CycleGAN과 아주 비슷하다. 유의해야 될 것만 나열.
1. 위의 그림처럼 기존의 CycleGAN의 Generator ($G_{AB}, G_{BA}$)와 Discriminator에 Attention Module (그림에서 $G_{AttnA}$, $G_{AttnB}$)를 붙임.
2. 생성 이미지는 attention mask를 사용해서 real과 raw gene이미지의 보간으로 들어간다. attention mask는 sigmoid를 통과하므로 0에서 1값을 가진다.
Attention Module과 Generator는 Discriminator를 속이기 위하여 1) 어떠한 관심 지역을 location하고, 2) located area에 올바르게 translation을 수행한다. 
이상적으로 수렴할 경우, Attention map은 점점 binary mask가 되고, 생성에 관심이 없는 부분은 0의 값을 갖는다. 
attention map으로 보간된 최종 생성 이미지를 식으로 나타내면 아래와 같다.
\begin{equation}
s^{\prime} = s_a \odot G_{A \rightarrow B}(s) + (1-s_a) \odot s
\end{equation}
앞의 항은 생성이미지로부터의 foreground부분, 뒤에는 실제 이미지로 부터의 background이다. horse2zebra를 예로 들면, 앞의 항은 말에서 생성된 얼룩말만 해당하고, 뒤의항은 말이미지의 배경이다. 

### Attention-guided Discriminator
Discriminator의 경우에는 Generator가 학습하면서 관심 영역을 지정하고 그 부분을 집중해서 변형시킨다. 이와 달리, 일반적인 discriminator는 전체부분을 다 보는데, 이는 실제 사진의 배경과 생성 사진의 foreground를 
모두 고려하는 것을 의미한다. 예를 들어, 말에서 얼룩말을 생성하는데 말에 대한 부분은 Generator가 완벽히 생성했다고 가정하자. 하지만 얼룩말은 얼룩말 사진의 실제 분포 (예를 들어 사바나), 말은 말 사진의 실제 분포 (목초지)을 따르므로 
이는 전체적인 관점에서는 생성된 이미지로 분류할 수 있다. 이 현상을 방지하기 위해 Generator는 전체 이미지를 모두 생성할 수 밖에없다. 즉, attention map은 모든 픽셀값이 1이 되고, 모델은 기존의 cyclegan이 된다. 실제로 논문의 supplementary 그림 2를 보면 epoch가 진행될수록 attention map이 밝아지는 것을 확인할 수 있다. 

따라서 논문에서는 discriminator가 전체를 보지 않도록 훈련하는 것을 제안하였다. 하지만 단순히 attention map과 생성 이미지를 element-wise곱 하는것은 초기에 학습되지 않은 attention map을 적용하기 때문에 오히려 학습이 
불안정해 질 수 있다. 따라서 논문에서는 30에폭까지 discriminator가 전체 이미지를 보고, 그 이후부터 attention mask를 적용하여 Discriminator를 학습한다. 

또하나의 문제점은 attention map이 항상 이상적으로 학습될 수 없다. 어느 작은 픽셀부분은 0이 아닌값을 가질 수 있기 때문에, 논문에서는 30에폭이 지난이후에 attention map에 threshold (논문에서는 0.1)을 사용하여 이 값밑의 값은 0으로 
변환하였다. 이것은 real이미지와 gene 이미지에 동시에 모두 적용된다. 하지만 실제 코드에서는 attention map을 곱하는 것은 구현되어 있지만, threshold를 적용하는 것은 구현되어 있지 않다. 또한 특정 에폭 이후에 attention module을 훈련하지 않는것은
구현되어 있으나 그 전에 Discriminator가 전체 이미지를 다 보는 것도 설정 안되어있다. 코드에서는 use_mask_for_D 변수를 True로 하면 D에 들어갈 때 attention map을 곱해서 들어가지만 threshold를 적용하지는 않고 에폭에 대한 추가적인 설정이 없어서 항상 attention map 곱해지거나 안곱해지거나 이다. 







