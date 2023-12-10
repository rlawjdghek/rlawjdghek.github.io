---
title: "[ICCV2021]Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
excerpt: "[ICCV2021]Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
categories:
  - Paper & Knowledge
  
tags:
  - Transformer
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-03-27T15:04:00-05:00
---
![](/assets/images/2022-03-27-SwinTransformer/9.jpg)
위의 그림은 SwinTransformerBlock overview

StyleGAN2를 트랜스포머로 구현한 StyleSwin을 공부하기 전 베이스가 된 논문. 내 구현은 [링크](https://github.com/rlawjdghek/Generative_Models/tree/main/GANs/Swin%20Transformer) 참조. 
구현해보고나니 이전에 리뷰한 TransGAN에서 이 모델을 거의 차용했었다. 하지만 TransGAN은 swin transformer를 크게 언급하지 않았음. Transformer의 유일한 단점인 parameter수를 조절하고,
패치간의 merging을 통하여 더 넓은 receptive field를 가지게 하였다.

### Abstract & Introduction
기존의 VIT는 224의 이미지로만 학습이 되었다. 트랜스포머의 구조적 특성상 고해상도의 이미지에 대하여 제곱으로 연산량이 증가하고, 파라미터갯수도 기하급수적으로 늘어난다.
본 논문에서는 이러한 기존의 ViT의 문제점을 해결하여 입력 이미지의 크기에 선형으로 비례하는 모델을 제시한다. 본 논문에서 제시한 shift window attention과 patch merging은 CNN의 locality bias를 
어느정도 갖게 하지만, 이것이 결과적으로 모델 성능의 향상에 도움이 된다는 것을 보여준다. 또한 transformer가 비전 분야에도 원활히 적용될 수 있도록 효율적인 연산량과 파라미터를 보여줌으로써
비전에서의 트랜스포머 사용의 시작을 알린다. 

![](/assets/images/2022-03-27-SwinTransformer/1.PNG)
### Method 
모델의 구조는 위의 그림과 같다. 가장 중요한 것은 3차원 이미지 데이터를 자연어에서와 같이 2차원으로 만들면서 [BS x C x H x W]가 [BS x H*W x C]가 되는데, 이 때 해상도가 커질수록 중간 차원이 
제곱연산으로 증가하여 행렬곱 연산에 큰 무리가 있다. 하지만 본 논문에서는 해상도가 절반으로 줄면서 입력 해상도에 robust한 모델구조를 제시한다. 기존의 ViT와 Swin Transformer의 연산량을 식으로 비교해보자.

#### Shifted Window based Self-Attention
self attention연산을 위해서 먼저 q,k,v의 차원을 [BS x H*W x C]라고 하자. 또한 [A x B], [B x C]의 크기를 가진 두 행렬의 행렬곱 연산은 ABC임을 기억하자.
1. Multi-head self attention<br/>
\begin{equation}
    attn = q k^T = H * W \times C \times H  *W = (H * W)^2 \times C
\end{equation}
\begin{equation}
    attn @ v = H * W \times H * W \times C = (H * W)^2 \times C
\end{equation}
따라서 논문 (1)번식의 두번째 항이 완성된다. 첫번째 항은 q,k,v를 계산하기 위한 연산량이다. 

2. Shifted Window based self attention
q와 k를 [BS x (M * M) x (H//M * W//M * C)]로 변형한 다음, 곱하면
\begin{equation}
    attn = q@k^T = (M * M) \times (H//M * W//M * C) \times (M * M) = M^2 \times H*W \times C
\end{equation}
이것도 마찬가지로 v까지 수행해준다면 논문의 (2)번식 두번째항과 같은 값이 나온다. 따라서 기존의 $(H * W)^2$의 연산량을 $H * W$로 줄일 수 있게 되었다.

위의 그림 (b)를 보면 한개의 swin transformer block은 W-MSA와 SW-MSA로 되어있다. 코드를 살펴보면, 
```python
def forward(self, x):
    H = W = self.in_res
    B, N, C = x.shape
    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)

    # x 를 shift 하는 것은 간단.
    if self.shift_size > 0: 
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1,2))
    else: 
        shifted_x = x
    x_windows = window_partition(shifted_x, self.window_size)
    x_windows = x_windows.reshape(-1, self.window_size**2, C)

    attn_windows = self.attn(x_windows, self.attn_mask)  # [(BS*가로*세로) x window**2 x C]
    attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, window_size=self.window_size, H=H, W=W)

    # 다시 역 shift로 맞춰준다.
    if self.shift_size > 0:
        x = torch.roll(shifted_x, (self.shift_size, self.shift_size), dims=(1,2))
    else:
        x = shifted_x

    x = x.reshape(B, N, C)
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return 
```
중간에 shift_size가 존재할 경우 (구현에서는 window_size//2 값을 사용함), 입력 feature map을 torch.roll함수를 통하여 shift_size만큼 shift 해준다. 그 뒤에 self.attn 레이어를 통하여 self-attention을 수행하는데, 
그 뒤에 다시 shift를 역으로 수행하여 본래의 이미지로 되돌아가는 것에 주의하자. 아래 그림 참고.
![](/assets/images/2022-03-27-SwinTransformer/3.PNG)



#### Patch merging
그 다음으로 연산량을 줄이는 것은 patch merging레이어이다. 이것도 간단하므로 코드를 보자.
```python
class PatchMerging(nn.Module):  # [BS x H*W x C] => [BS x H//2 x W//2 x 4C]
    def __init__(self, dim, in_res, norm_type="ln"):
        super().__init__()
        self.in_res = in_res
        assert self.in_res % 2 == 0
        self.norm_layer = CustomNorm(norm_type, 4*dim)
        self.reduction_layer = nn.Linear(4*dim, 2*dim, bias=False)
    def forward(self, x):  # x : [BS x H*W x C]
        BS, N, C = x.shape
        assert N == self.in_res ** 2
        x = x.reshape(BS, self.in_res, self.in_res, C)
        x1 = x[:, 0::2, 0::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, :]
        x = torch.cat([x1,x2,x3,x4], dim=-1)  # [BS x H//2 x W//2 x 4C]
        x = x.reshape(BS, -1, 4*C)
        
        x = self.norm_layer(x)
        x = self.reduction_layer(x)
        return x  
```
입력을 [BS x H x W x C]로 reshape한 뒤, 아래 그림처럼 H//2, W//2크기의 feature map 4개로 나눈다. 
![](/assets/images/2022-03-27-SwinTransformer/2.PNG)

그 다음 채널을 기준으로 concatenation 하는데, 이러면 채널이 4배로 늘어나므로 다시 C로 reduction 해준다. 이로 인하여 [BS x H x W x C] => [BS x H//2 x W//2 x C]로 해상도를 4배 낮출수 있다. 

#### Efficient Batch Computation for Shifted Configuration
shift 될 경우 attnetion연산을 window내에서만 해야하기 때문에 attention mask를 특수한 attention mask를 사용한다. 
![](/assets/images/2022-03-27-SwinTransformer/6.PNG)
위의 그림을 보면, shift할 것을 고려하여 window partitional을 해준다. 그러면 그 왼쪽의 그림과 같이 shift 된 A,B,C 구역은 회색 구역과 attention 연산이 겹치면 안된다. 따라서 만드는 것이 아래 코드의 attn_mask이다. 아래 코드를 이해하기 위해서는 많은 그림이 필요하므로 이미지로 첨부한다. 
```python
 if self.shift_size > 0:
            H, W = in_res, in_res
            img_mask = torch.zeros((1,H,W,1))
            h_slices = [
                slice(0, -window_size), 
                slice(-window_size, -shift_size), 
                slice(-shift_size, None)
            ]
            w_slices = [
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None)
            ]
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # [(가로*세로) x window_size x window_size x 1]
            mask_windows = mask_windows.reshape(-1, self.window_size**2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill_(attn_mask!=0, -100.0).masked_fill_(attn_mask==0, 0.0)
```
![](/assets/images/2022-03-27-SwinTransformer/7.PNG)
![](/assets/images/2022-03-27-SwinTransformer/8.PNG)

#### Architecture variants
본 논문에서는 유연한 사이즈에 대하여 4가지의 모델을 실험하였다. 기존 ViT 단점중 하나는 모델이 큰만큼 큰 데이터셋에서만 잘 작동하고, 훈련 시간도 아주 길었다는 것인데, swin transformer는 
ViT보다 훨씬 더 가벼우면서 좋은 성능을 자랑한다. 
논문에서 제시하는 4가지 모델은 아래 표와 같은 구조를 가진다. 
![](/assets/images/2022-03-27-SwinTransformer/4.PNG)

### Experiment
새롭게 제시된 swin transformer는 여러 비전 task에서 우수한 성능을 보인다는 것을 입증하기 위하여 본 논문에서는 컴퓨터 비전의 대표적인 task 3개, image classification, object detection, segmentation에 
대하여 기존 모델들과 비교한다. 
![](/assets/images/2022-03-27-SwinTransformer/5.PNG)

위의 이미지넷 결과에서 볼 수 있듯이, 기존 ViT는 86M, 307M의 파라미터를 가지지만, Swin-T 같은 경우 파라미터는 3배이상, FLOPs는 10배 이상 효율적이다. 또한 성능도 80대를 넘는 것을 보여준다. 즉, 이미지넷에서 기존 CNN과 
대등한 모델의 연산량을 갖고, 성능도 우수한 Transformer를 최초로 제시하였다. 


