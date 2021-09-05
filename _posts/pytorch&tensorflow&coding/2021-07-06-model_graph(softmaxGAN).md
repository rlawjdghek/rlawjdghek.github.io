---
title:  "Pytorch 모델 그래프 구조"
excerpt: "SoftmaxGAN에서의 특별한 구조"
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

아래 그림은 SoftmaxGAN의 그래프이다. 이대로 구현 하면 되는데 이전의 GAN들과 다른점은 Generator를 훈련시키기 위해 real logit도 사용한다는 점이다.
**순서에 주의하지. 먼저 D를 업데이트 시키고 업데이트 된 D에서 logit을 추출하여 G_loss를 만든다. 또한 D를 훈련시킬때에는 optimizer_D를 사용하고, 
G는 optimizer_G를 사용하므로 그래프가 연결되어 있어도 업데이트는 각각 될수 밖에 없다. 당연한 얘기지만, optimizer_D.step을 하면 D의 parameter만 된다.**
![](/assets/images/2021-07-06-model_graph(softmaxGAN)/1.JPG)

구현하기 위해 아래 코드를 보자. 
```python
for epoch in range(args.n_epochs):
    for idx, (img, target) in enumerate(train_loader):
        img = img.cuda()
        B = 1/(args.batch_size*2)
        latent_z = torch.FloatTensor(np.random.randn(args.batch_size, args.n_latent)).cuda()
        gene_img = G(latent_z)
        
        real_logit = D(img)
        gene_logit = D(gene_img)
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))  # Z_B는 모든 배치에 대하여 구하는것.
        
        # training D        
        D_loss_ = torch.mean(real_logit) + torch.log(Z_B)
        optimizer_D.zero_grad()
        D_loss_.backward(retain_graph=True)
        optimizer_D.step()

        # training G
        G_loss_ = torch.sum(real_logit) * B + torch.sum(gene_logit) * B + torch.log(Z_B)
        optimizer_G.zero_grad()
        G_loss_.backward()
        optimizer_G.step()        
```
위의 코드는 D를 훈련 시키고 retain_graph=True로 하여 그래프를 남긴 뒤, G_loss_를 계산하기 위해 처음에 계산한 real_logit과 gene_logit을 사용한다. (Z_B도 마찬가지).
그런데 문제는 파이토치에서 parameter가 업데이트 되는 원리는 inplace연산으로 (메모리를 아끼기 위해 사용), 모델과 값들이 가만히 있는 것 처럼 보인다. 예를 들어 x += 2.
그래서 D의 훈련에 사용되었던 real_logit과 gene_logit을 사용하고, optimizer_G.step에서 이 사용된 logit들이 들어가기 때문에 optimizer_G.step()에서 오류가 뜨는 것이다.
따라서 아래 3가지로 해결 할 수 있다.

1. 그냥 gene_logit과 real_logit을 새로 만들기. detach사용
```python
for epoch in range(args.n_epochs):
    for idx, (img, target) in enumerate(train_loader):
        img = img.cuda()
        B = 1/(args.batch_size*2)
        latent_z = torch.FloatTensor(np.random.randn(args.batch_size, args.n_latent)).cuda()
        gene_img = G(latent_z)
        
        real_logit = D(img)
        gene_logit = D(gene_img)
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))  # Z_B는 모든 배치에 대하여 구하는것.
        
        # training D        
        D_loss_ = torch.mean(real_logit) + torch.log(Z_B)
        optimizer_D.zero_grad()
        D_loss_.backward()
        optimizer_D.step()

        # training G
        real_logit = D(img)
        gene_logit = D(gene_img)
        G_loss_ = torch.sum(real_logit) * B + torch.sum(gene_logit) * B + torch.log(Z_B)
        optimizer_G.zero_grad()
        G_loss_.backward()
        optimizer_G.step()  
```  
이 때, D를 훈련시키는 부분에서 gene_logit을 생성하는 부분에서 detach를 사용하였기 때문에 G가 연결되지 않는것을 이용.

2. 1.과 마찬가지로 real과 gene에 forward 2번씩. retain_graph=True 사용
```python
for epoch in range(args.n_epochs):
    for idx, (img, target) in enumerate(train_loader):
        img = img.cuda()
        B = 1/(args.batch_size*2)
        latent_z = torch.FloatTensor(np.random.randn(args.batch_size, args.n_latent)).cuda()
        gene_img = G(latent_z)
        
        real_logit = D(img)
        gene_logit = D(gene_img)
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))  # Z_B는 모든 배치에 대하여 구하는것.
        
        # training D        
        D_loss_ = torch.mean(real_logit) + torch.log(Z_B)
        optimizer_D.zero_grad()
        D_loss_.backward(retain_graph=True)
        optimizer_D.step()

        # training G
        real_logit = D(img)
        gene_logit = D(gene_img)
        G_loss_ = torch.sum(real_logit) * B + torch.sum(gene_logit) * B + torch.log(Z_B)
        optimizer_G.zero_grad()
        G_loss_.backward()
        optimizer_G.step()  
```  

3. 먼저 D_loss, G_loss 구한 뒤 backward
세 번쨰 방법이 특이한데, 먼저 D_loss와 G_loss를 계산한뒤 한번에 업데이트 한다. 코트 확인.
```python
for epoch in range(args.n_epochs):
    for idx, (img, target) in enumerate(train_loader):
        img = img.cuda()
        B = 1/(args.batch_size*2)
        latent_z = torch.FloatTensor(np.random.randn(args.batch_size, args.n_latent)).cuda()
        gene_img = G(latent_z)
        
        real_logit = D(img)
        gene_logit = D(gene_img)
        Z_B = torch.sum(torch.exp(-real_logit)) + torch.sum(torch.exp(-gene_logit))  # Z_B는 모든 배치에 대하여 구하는것.
        D_loss_ = torch.mean(real_logit) + torch.log(Z_B)
        G_loss_ = torch.sum(real_logit) * B + torch.sum(gene_logit) * B + torch.log(Z_B)

        if idx % 2 == 1:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()
            D_loss_.backward(retain_graph=True)
            G_loss_.backward()
            optimizer_D.step()
            optimizer_G.step()
        
        if idx % 1 == 0:
            optimizer_G.zero_grad()
            G_loss_.backward()
            optimizer_G.step()
```         
실제 구현에서는 D 1번에 G 2번을 훈련했다. 이렇게 안하고 1번 1번 하면 검정색 사진만 나옴. 