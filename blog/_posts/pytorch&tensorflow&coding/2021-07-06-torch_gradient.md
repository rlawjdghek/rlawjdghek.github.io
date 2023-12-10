---
title:  "torch gradient 계산 정리"
excerpt: "torch gradient 계산 정리 (WGAN)"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - Pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T12:48:00-05:00
---

먼저 gradient를 할때에 주의할 점.
```python
input1 = torch.FloatTensor([3.0]).requires_grad_(True)
input2 = torch.FloatTensor([2.0]).requires_grad_(True)
```
만약 FloatTensor만 선언한다면 requires_grad는 False로 되어있어서 backward에서 오류가 난다. 무조건 지정. 단 레이블 같이 학습이 필요없는것은 안해도 됨.
그러고 곱셈을 하면, 이 때에는 input1의 grad 는 None이다. 그 다음에 backward를 실행하면 c의 grad_fn은 multiplication으로 되고,
input1, input2의 grad가 정해진다.

### torch.autograd.grad
WGAN-GP를 구현하면서 사용한 함수이다. 역할은 그래디언트를 직접적으로 텐서로 볼 수 있는 함수이다. 그래디언트를 직접계산해보거나 그래디언트가 손실 함수로써 사용될 때 
사용한다. 아래 코드를 보자. 
```python
p_real = torch.randn((4,1,28,28)).cuda()
p_gene = torch.randn(real_sample.shape).cuda()

D = Discriminator((1,28,28)).cuda()
alpha = torch.rand((p_real.shape[0], 1, 1, 1)).cuda()
interpol = (alpha * p_real + (1 - alpha) * p_gene).requires_grad_(True)  # p_real과 p_gene의 interpolation한 값.
interpol_logit = D(interpol)
ones_label = torch.ones(interpol_logit.shape).requires_grad_(False).cuda()

g = torch.autograd.grad(
    outputs=interpol_logit,
    inputs=interpol,
    grad_outputs=ones_label, 
    retain_graph=True,  # 그래프가 여러번 backward되어도 안사라짐.
    create_graph=True,  # 그래프를 생성해야 이것도 로스로서 유효하게 역전파에 영향을 줄 수 있다.
    only_inputs=True
)[0]

twos = torch.zeros_list(interpol_logit).fill_(2.0)
g2 = torch.autograd.grad(
    outputs=interpol_logit,
    inputs=interpol,
    grad_outputs=twos,
    retain_graph=True,
    create_graph=True,
    only_inputs=True
)
```
위의 코드는 실제로 WGAN-GP에 쓰인 gradient penalize (GP)항을 계산하기 위한 코드다. 그 중 torch.autograd.grad함수만 보면,
* outputs: 최종 결과물.
* inputs: outputs을 이걸로 미분할 것이다.
* grad_ouputs: 그냥 최종 결과물에 이 텐서를 곱하는 것으로 이해함. 이게 값이 2배가 되면 미분 값도 2배가 되서 나온다. 
* retain_graph: 이게 False로 되어있으면 한번 backward하면 그래프가 없어짐. True이면 그래프가 여러번 backward되어도 안사라진다.
* create_graph: 원래 이게 먼저 나와야 맞지 않나 싶다. 그래프를 생성해야 이것도 최종 로스에 들어가는 것이 의미가 있다. 
즉, 이 gradient도 그래프로 만들어야 역전파에 관여되면서 추가적인 fine-tune이 이루어 질 수 있다.
* only_inputs: 잘 모르겟당. 

