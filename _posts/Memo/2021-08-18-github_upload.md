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

 

깃허브의 레포지토리의 주소는 https://github.com/rlawjdghek/git_test이고 

C:\Users\rlawjdghek\git\에 test.py를 추가할 것이라고 가정하자. 

이 test.py를 위의 레포지토리 주소에 올려야 한다.

 

1. 먼저 깃헙과 내 로컬컴퓨터를 동기화 시켜야 한다. 

C:\Users\rlawjdghek\git 에서 git bash here 를 한다. git clone https://github.com/rlawjdghek/git_test 하면 git_test라는 폴더가 통째로 날라온다. 즉 git\git_test가 만들어 진다. 나갔다가 git_test에서 마우스로 git bash here로 다시 킨다. 이후에 git status하면 완전히 동기화 되어있으므로 nothing to commit이라고 뜬다. 

 

2. 수정한 파일을 이 폴더에 올린다. 그 다음에 아래 명령어

git add * 하면 여기에 있는 모든 파일이 staging area로 올라간다.  

 

3. git commit -m "asdasd"

 

4. git push origin master 

 

* 100MB 이상 파일은 못올리므로 포함 안되게 주의하자