---
title:  "Torch tensors"
excerpt: "torch.Tensor, torch.tensor, torch.autograd.Variable"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T14:48:00-05:00
---

torch.Tensor는 그냥 상수라고 보면 된다. 즉 gradient를 계산 할 수 없다. 따라서 아래 예시처럼 gradient 를 계산 하려고 하면 에러가 뜬다. requred grad가 없음. 
![](/assets/images/2021-07-06-torch_tensors/1.JPG)

하지만 torch.tensor는 매개변수에 requires_grad가 있다. 또한 이 변수를 사용 하려면 무조건 변수가 소수여야하는 것을 주의하자. 즉 torch.tensor([1,2,3,4,5], dtype = torch.float32, requires_grad = True)로 하면 torch.autograd.Variable한 것과 똑같은 효과를 얻는다

 

마지막으로 torch.autograd.Variable은 그냥 wrapper의 개념으로 쓴다. 무조건 requires_grad를 true로 할수있는 wrapper. 따라서 torch.Tensor도 torch.tensor로 만들 수 있다. 
아래의 두 표현은 같다. 
```python
torch.autograd.Variable(torch.Tensor([1,2,3,4]), requires_grad = True) 
== torch.tensor([1,2,3,4], dtype = torch.float32, requires_grad = True)
```
 

따라서 여러 방법이 있는데도 굳이 Variable을 써서 길게 나타내는 이유는 어차피 이걸 쓰는 변수들은 한번만 선언 하면 되는거고 특수한 목적이 있는것만 가끔씩 쓰는 것이므로 강조하기 위해서 쓴다. 
예를 들어 GAN훈련에서 진짜 가짜의 레이블을 나타낼 때 쓰는 \
```python
ones_label = torch.autograd.Varialbe(torch.ones(BATCH_SIZE), requires_grad = False).to(device)
```
같은 경우는 그냥 torch.ones안에서 requies_grad=False를 해도 되는데 그냥 강조하려고 쓰는 것이다. 