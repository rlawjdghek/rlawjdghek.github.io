---
title:  "GAN 훈련 G부터? D부터?"
excerpt: "속도차이가 날 때가 있다."
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

2021-05-31 : 정석대로라면 보통 D부터 훈련하는게 맞다고 들었다. 그런데 BEGAN을 훈련하다보니 훈련속도가 너무 느려서 오피셜 코드와 비교하다 보니 흥미로운 사실을 발견함. 

아래 코드처럼 모든 코드가 같고 훈련 순서만 바꿔서 훈련해보았는데 G부터 훈련하는것이 약 8배정도 더 빠르다. 

빠른 코드
```python
for epoch in range(argsn.n_epochs):
    kt = args.k0
    gamma = args.gamma
    G_loss = 0
    D_loss = 0
    train_loop = tqdm(train_loader, total=len(train_loader))
    for n, img in enumerate(train_loop):
        img = img.cuda()
        latent_z = Variable(torch.randn(args.batch_size, args.lantet_dim), requires_grad=False).cuda()
        gene_img = G(latent_z)
        
        # training G
        recons_gene-img = D(gene_img)
        G_loss_ = criterion_L1(gene_img, recons_gene_img)
        optimizer_G.zero_grad()
        G_loss_ = backward()
        optimizer_G.step()
    
        # training D
        recons_img = D(img)
        recons_gene_img = D(gene_img.detach())
        D_loss1 = criterion_L1(img, recons_img)
        D_loss2 = criterion_L1(gene_img.detach(), recons_gene_img)
        D_loss_ = D_loss1 + D_loss2
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # calc kt
        start = time.time()
        diff = gamma * D_loss1 - G_loss_
        t1 = time.time()
        kt += args.k_lr * diff.item()
        t2 = time.time()
        kt = min(max(kt, 0), 1)
        t3 = time.time()
        
        G_loss += G_loss_.item()
        D_loss += D_loss_.item()

    print(f"Epoch: [{epoch}/{args.n_epochs}] | G loss: {G_loss / len(train_loader)}, D loss: {D_loss / len(train_loader)}")
```

느린 코드
```python
for epoch in range(argsn.n_epochs):
    kt = args.k0
    gamma = args.gamma
    G_loss = 0
    D_loss = 0
    train_loop = tqdm(train_loader, total=len(train_loader))
    for n, img in enumerate(train_loop):
        img = img.cuda()
        latent_z = Variable(torch.randn(args.batch_size, args.lantet_dim), requires_grad=False).cuda()
        gene_img = G(latent_z)
    
        # training D
        recons_img = D(img)
        recons_gene_img = D(gene_img.detach())
        D_loss1 = criterion_L1(img, recons_img)
        D_loss2 = criterion_L1(gene_img.detach(), recons_gene_img)
        D_loss_ = D_loss1 + D_loss2
        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # training G
        recons_gene-img = D(gene_img)
        G_loss_ = criterion_L1(gene_img, recons_gene_img)
        optimizer_G.zero_grad()
        G_loss_ = backward()
        optimizer_G.step()

        # calc kt
        start = time.time()
        diff = gamma * D_loss1 - G_loss_
        t1 = time.time()
        kt += args.k_lr * diff.item()
        t2 = time.time()
        kt = min(max(kt, 0), 1)
        t3 = time.time()
        
        G_loss += G_loss_.item()
        D_loss += D_loss_.item()

    print(f"Epoch: [{epoch}/{args.n_epochs}] | G loss: {G_loss / len(train_loader)}, D loss: {D_loss / len(train_loader)}")
```

* 또한 DCGAN에서는 위의 경우가 거의 해당하지는 않았지만 G부터 훈련하는것이 약 10퍼센트정도 더 빨랐다.