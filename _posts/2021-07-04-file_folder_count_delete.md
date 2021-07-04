---
title:  "파일 / 폴더 갯수 세기, 삭제"
excerpt: "파일 / 폴더 갯수 세기, 삭제"
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

#### 현제 디렉토리에서 파일 갯수 세기
ls -l | grep ^- | wc -l

 

#### 현재 디렉토리에서 하위 모든 파일갯수 세기 
find .  -type f | wc -l

 

#### 현재 디렉토리에서 디렉토리 갯수세기
ls -l | grep ^d | wc -l

 

####현재 디렉토리에서 특정 폴더 안의 모든 파일 갯수 세기
만약 현재 디렉토리에 laplace_2, laplace_5, laplace_8 폴더가 있다고 할 때 이 폴더 안의 모든 파일 갯수를 센다. 
예전에는 하나씩 들어가서 셋는데 지금은 현재 디렉토리에서 아래 명령어를 입력

ls -R laplace_* | wc -l 

 
#### 일부 규칙으로 빼기
tensordecomp하던 중 생성된 데이터가 꼬이면 일정 폴더에 있는 일부 파일들을 규칙성을 이용해 빼내거나 삭제해야한다.
예를 들어, real이미지는 5글자.npy이고 fake이미지는 6글자.npy일때 real폴더에 fake파일이 들어가서 지우려고 한다 해보자. 
그러면 와일드 카드를 써서 6글자인 파일들을 모두 선택해야 되는데 아래와 같이 할  수 있다.

rm -r ??????.npy 하면 파일 이름이 6글자인 파일은 모두 삭제된다. 