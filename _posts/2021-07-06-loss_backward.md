---
title:  "loss.backward 원리 메모"
excerpt: "retain graph=True의 필요성 & GAN 학습할때"
categories:
  - Pytorch & Tensorflow & Coding
  
tags:
  - GAN
  - pytorch
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-06T14:48:00-05:00
---

가장 기본적인 GAN을 학습한다고 해보자. D를 먼저 훈련, 그 다음에 G를 훈련 alternative 하게
1. latent_z를 만들고 gene_img를 만들어서 real_img와 비교해서 discriminator를 학습한 뒤에, 다시 latent_z를 만들고 gene_img만들어서 generator를 학습. => 한번의 에폭에 서로다른 gene_img로 학습시킴. forward도 두번. 안좋음
2. discirminator를 훈련하는 것은 1과 동일. latent_z를 만들지 않고 기존의 latent_z로 gene_img를 만들어서 학습시킴. => generator를 두번 돌려야 되므로 마찬가지로 시간 아까움.
3. 가장 정석적인 방법. detach를 활용해서 D에 gene_img부터 그래프를 끊는다. 아래 그림을 참조하자.
![](/assets/images/2021-07-06-loss_backward/2.JPG) 
detach가 없다면 D가 G가 만든 gene_img까지 받아오므로 D가 backward되면 G까지 업데이트 된 후에 그래프가 없어진다. G를 그래프에서 살리기 위해 
아래 코드와 같이 detach로 끊을 수 있다.
```python
ones_label = torch.autograd.Variable(torch.ones((BATCH_SIZE, 1)), requires_grad=False).cuda()
zeros_label = torch.autograd.Variable(torch.zeros((BATCH_SIZE, 1)), requires_grad=False).cuda()

for epoch in range(NUM_EPOCHS):
    for iter_, (img, _) in enumerate(train_loader):
        real_img = img.to(device)
        latent_z = nn.init.normal_(torch.zeros((BATCH_SIZE, LATENT_DIM))).to(device)
        gene_img = G(latent_z)

        # training D        
        real_preds = D(real_img)
        gene_preds = D(gene_img.detach())
        D_loss_real = criterion_CE(real_preds, ones_label)
        D_loss_gene = criterion_CR(gene_preds, zeros_label)
        D_loss_ = (D_loss_real + D_loss_gene) / 2
        optimizer_D.zero_grad()
        D_loss_.backward()
        optimizer_D.step()

        # training G
        gene_preds = D(gene_img)
        G_loss_ = criterion_CE(gene_preds, ones_label)
        optimizer_G.zero_grad()
        G_loss_.backward()
        optimizer_G.step()
```

4. retain_graph=True로 함으로써 그래프를 살릴 수 있다. 지금 문제는 D가 먼저 학습되면서 그래프를 없애는 것인데 D가 학습한 뒤에 그래프를 온전히 남김으로써 G가 학습할때에도 정상적으로 backward가 가능하다.

* 아래코드는 안되는 예제이다.
```python
ones_label = torch.autograd.Variable(torch.ones((BATCH_SIZE, 1)), requires_grad=False).cuda()
zeros_label = torch.autograd.Variable(torch.zeros((BATCH_SIZE, 1)), requires_grad=False).cuda()

for epoch in range(NUM_EPOCHS):
    for iter_, (img, _) in enumerate(train_loader):
        real_img = img.to(device)
        latent_z = nn.init.normal_(torch.zeros((BATCH_SIZE, LATENT_DIM))).to(device)
        gene_img = G(latent_z)

        # training D        
        real_preds = D(real_img)
        gene_preds = D(gene_img)
        D_loss_real = criterion_CE(real_preds, ones_label)
        D_loss_gene = criterion_CR(gene_preds, zeros_label)
        D_loss_ = (D_loss_real + D_loss_gene) / 2
        optimizer_D.zero_grad()
        D_loss_.backward()
        optimizer_D.step()

        # training G
        gene_preds = D(gene_img)
        G_loss_ = criterion_CE(gene_preds, ones_label)
        optimizer_G.zero_grad()
        G_loss_.backward()
        optimizer_G.step()
```
 
2021-04-05: 이상하게도 몇몇 GAN에서는 G부터 학습하는 것이 빠르다. (거의 모든 경우 빠르거나 비슷하다.) 그래서 GAN을 구현 한 코드를 보면 종종 G부터 훈련하는 경우가 있다. 
나도 특별한 경우 아니면 G부터 훈련하도록 코드를 구현한다. 