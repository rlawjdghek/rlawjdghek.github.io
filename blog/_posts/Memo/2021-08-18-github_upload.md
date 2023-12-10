---
title:  "깃헙 간단 글 올리기"
excerpt: "깃헙 간단 글 올리기"
categories:
  - Memo
  
tags:
  - github
  
toc: true
toc_sticky: true
toc_label: "On this page"
use_math: true
    
last_modified_at: 2021-08-18T14:48:00-05:00
---

작업 폴더 -> staging area -> repository 순으로 파일이 간다고 보면 된다. 즉, 깃헙에 파일이 올라갈때 먼저 staging area에 add하고 commit 후 repository로 push한다. 

1. 깃헙에서 저장소(폴더) 생성
2. 로컬에 저장소(폴더) 생성
3. git bash here로 키고 git init
4. 로컬 저장소에 파일 추가 (git add)
5. 로컬 저장소에 커밋 (git commit)
6. 브렌치 이름변경 (git branch -M main)
7. 깃허브 원격 저장소 추가 (git remote add origin)
8. 깃허브 코드 업로드 (git push)

예를 들어보면,
1. 깃헙 페이지 가서 New Repository -> git_test 폴더 생성
2. D:/git/git_test 생성 후, asdasd.txt 파일을 만들었다고 가정.
3. git_test에서 git base here 후 git init
4. git add .
5. git commit -m "git test upload"
6. git branch -M main
7. git remote add origin https://github.com/rlawjdghek/git_test
8. git push -u origin main

* 100MB 이상 파일은 못올리므로 포함 안되게 주의하자