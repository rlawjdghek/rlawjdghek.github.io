---
title:  "Convolution Layer"
excerpt: "Convolution Layer"
categories:
  - Paper & Knowledge
  
tags:
  - Paper & Knowledge
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-14T15:33:00-05:00
---

이 글을 쓰는 시점은 ICCV KD 하다가 SepConv함수에서 

```python
self.op = nn.Sequential(
    nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
    nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
    nn.BatchNorm2d(channel_in, affine=affine),
    nn.ReLU(inplace=False),
    nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
    nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
    nn.BatchNorm2d(channel_out, affine=affine),
    nn.ReLU(inplace=False),
)

```
가 나옴. 아래 내용을 보면 이는 DepthWise Separable Convolution을 2번 썼다는 것을 알 수 있다. 

### Depthwise Convolution Layer
![](/assets/images/2021-08-14-Convolution_Layer/1.PNG)
위 그림처럼 각각의 입력 채널을 나누어서 convolution 연산을 진행한다. 필요한 커널은 입력 채널의 갯수만큼 필요하다. 따라서 위 그림과 같이 입력 채널과 출력 채널이 같다. 장점으로는 연상량이 훨씬 적어진다는 것. 

### PointWise Convolution Layer
1x1 Convolution을 생각하면 된다. 차원을 조절할 때 많이 쓰인다. 
![](/assets/images/2021-08-14-Convolution_Layer/2.PNG)

### DepthWise Separable Convolution Layer
Depthwise + Pointwise convolution.