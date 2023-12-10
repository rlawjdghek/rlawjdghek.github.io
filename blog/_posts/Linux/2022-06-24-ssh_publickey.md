---
title:  "윈도우에서 Public Key로 리눅스 ssh 접속 비밀번호 입력 패스하기."
excerpt: "윈도우에서 Public Key로 리눅스 ssh 접속 비밀번호 입력 패스하기."
categories:
  - Linux
  
tags:
  - Linux
 
published: True
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2022-06-24T15:33:00-05:00
---

1. 로컬에 C:Users\rlawjdghek\.ssh\id_rsa.pub을 서버에 복사
2. mkdir ~/.ssh
3. touch ~/.ssh/authorized_keys
4. chmod 755 ~/.ssh/authorized_keys
5. cat ./id_rsa.pub >> ~/.ssh/authorized_keys


