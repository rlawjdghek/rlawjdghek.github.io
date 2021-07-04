---
title:  "백그라운드 파일 실행"
excerpt: "nohup활용"
categories:
  - Linux
  
tags:
  - Linux
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

무언가를 출력하는 파이썬 파일을 실행하고 싶을때, 보통은 터미널에서 

```bash
python asdasd.py
```
를 실행하면 네트워크 연결이 끊기거나 다른 이유가 있어서 터미널이 꺼지는 경우 프로세스도 자동으로 꺼진다. 그럴 때를 대비해서 오래 돌아가는 프로세스는 백그라운드로 실행하는 것이 편하다. 

 
```bash
nohup python asdasd.py &
```
을 실행하면 백그라운드에서 돌아가면서 현재 경로에 nohup.out 이 생성된다. 근데 보통 out파일의 이름을 정해주니까 아래와 같은 명령어를 쓰자

```bash
nohup python asdasd.py &> asdasd.out &
```
그러면 nohup.out이 생기는 것이 아니라 asdasd.out이 생긴다. 끌때는 kill로 끈다.

 

출력문을 볼 때에는 

```bash
tail asdasd.out
```
을 하면 지금까지 출력된 모든 파일을 볼 수 있고,

```bash
tail -f asdasd.out
```
하면 실시간으로 볼 수 있다. (빠르게 올라갈 때 좋음)**