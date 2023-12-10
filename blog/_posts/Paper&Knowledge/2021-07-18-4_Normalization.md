---
title:  "4가지 Normalization 기법"
excerpt: "Instance, Batch, Layer, Group Normalization"
categories:
  - Paper & Knowledge
  
tags:
  - Paper & Knowledge
 
published: false
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-18T23:19:00-05:00
---

결과부터 말하자면,\
Instance는 한개의 배치, 한개의 채널 기준. -> batch and channel-wise\
Batch는 모든 배치에 대하여 한개의 채널로 묶음. -> channel-wise\
Layer는 한개의 배치에 대하여 모든 채널로 묶음. -> batch-wise\
Group은 한개의 배치에 대하여 몇개의 채널로 묶음.\
\
### Batch Normalization
**안정적으로 수렴할 수 있다. 수렴속도가 빨라진다.** 구글링으로 검색하면 나오는 식과 실제로 파이토치의 계산법이랑 다르다. 아래의 그림과 같이 나옴.
![](/assets/images/2021-07-18-4_normalization/1.JPG)
그림을 보면 위의 배치 정규화의 결과가 나오기 위해서는 평균의 사이즈가 [3x2x2]가 나오는 것이 아니라, [3, ]이 나와야 한다. 즉, 같은 배치 위치에 있는 0,1,2,3과 
12, 13, 14, 15를 이용한다고 할 때, 만약 구글링에 나온 식이 맞다면 배치정규화의 결과는 그냥 -1과 1만으로 되어 있어야한다. (즉, 배치를 기준으로 element-wise하게 0과 12의 평균은 6이고
거리가 같으므로 정규화 계산하면 -1과 1이 나온다.) 하지만 실제로 값을 유도하기 위해서는 0, 1, 2, 3, 12, 13, 14, 15를 하나로 묶어서 평균 7.5와 표준편차 6.1032778를 이용해서 구하면 된다. 즉 
채널까지 고려. 아래 코드가 구현 한것.

```python
N, C, H, W = x.shape
mean = np.mean(x, axis=(0,2,3))
variance = np.maxn((X-mean.reshape(1,C, 1, 1)) ** 2, axis=(0, 2, 3))
x_hat = (x - mean.reshape((1, C, 1, 1)) * 1.0 / np.sqrt(variance.reshape(1, C, 1, 1)) + eps)
out = gamma.reshape((1, C, 1, 1)) * x_hat + beta.reshape((1, c, 1, 1))
``` 

### Instance Normalization
styleGAN에서 사용한 정규화. 채널별이라는 것을 명심하자. 배치 정규화에서 배치의 개념을 뺐다고 생각. 즉, 한개의 채널 단위로 정규화 한다고 본다. 결과는 아래와 같다.
![](/assets/images/2021-07-18-4_normalization/2.JPG)
asd의 값을 안 바꿔 줬을 땐 위에서 2번째 [2x2]행렬이 나머지와 같았는데 5를 20으로 늘려주니 그 행렬만 바뀐다.

### Layer Normalization
한개의 배치 단위로 본다. 즉 위에 있는 두개는 채널 단위이고 이건 모든 채널인데 한개의 배치.
![](/assets/images/2021-07-18-4_normalization/3.JPG)
위에 계산하려면 1번 배치 : 0, 1, ..., 11을 평균 구하고 표준편차 구하면 된다.

### Group Normalization
이건 instance와 Layer사이인데 몇개의 채널을 선택할 것인가를 설정. 즉, 이 값이 1이면 Instance이고 모든 채널이면 Layer이다. 이건 논문에 나중에 나오면 그 떄 공부하자.
