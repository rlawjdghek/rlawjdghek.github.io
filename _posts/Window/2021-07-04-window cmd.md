---
title:  "윈도우 cmd 복사, 삭제 명령어, 디렉토리별 크기 명령어"
excerpt: "윈도우 cmd 복사, 삭제 명령어, 디렉토리별 크기 명령어"
categories:
  - Window
  
tags:
  - Window
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

#### 추가

xcopy .\asdasd D:\ /e /h /s

/e : 하위 디렉토리까지 전부 복사

/s : 비어있지 않은 하위 디렉토리만 복사

/h : 숨겨진 파일과 시스템까지 복사

#### 삭제

rd .\asdasd /s /q

/s : 하위까지 모두 삭제

/q : 삭제할거냐는 말을 붙지 않음

또는 rmdir /s <directory name>

(또다른 예제)del /s /q jupyter

#### 디렉토리 크기 명령어

du.exe -q -l 1 d:\WebHosting\LocalUser