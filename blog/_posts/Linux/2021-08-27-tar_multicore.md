---
title:  "멀티코어로 압축 진행하기"
excerpt: "tar"
categories:
  - Linux
  
tags:
  - Linux
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-27T22:33:00-05:00
---

### 1. pizip2 설치
```bash
sudo apt-get install pbzip2
```

### 2. 압축 진행
```bash
tar --use-compress-prog=pbzip2  -cvf <압축 파일 이름> <압출할 파일>
``` 

예를 들어, VFP290K를 VFP290K.tar.bz2 로 잡축한다 하면
```bash
tar --use-compress-prog=pbzip2 -cvf VFP290K.tar.bz2 VFP290K
```

### 3. 압축 해제
```bash
tar -I lbzip2 -xvf <file.tar.bz2>
```
* 두 번째 매개변수는 대문자 "아이" 이다.