---
title:  "파이참 환경설정"
excerpt: "파이참 기본 설정 및 기록들"
categories:
  - configuration
  
tags:
  - configuration

last_modified_at: 2021-06-28T10:48:00-05:00
---

1. new project
2. setting에서 project: 에서 project interpreter에서 project interperter 옆에 add
3. SSH interpreter에서 Existing server configuration에서 rlawjdghek
4. 맨 윗줄에 connected to jeonghokim 가 있어야한다. 그 밑에 interpreter에서 원하는 가상환경 (예를 들어 .conda/faceswap/bin/python)
5. 그 밑에 sync folder에서 내가 만드는 코드들이 어디에 저장되는지 설정.
6. path mapping도 확인하기
7 . (중요) local path는 W:같이 붙여야하고 , 
remote path는 붙이면 안됨 그냥 /home/...이런식으로

위에건 기본 설정이고, 만약 내가 현재 폴더에서 파이참 프로젝트를 만들고 싶다. 
1. 오른쪽 클릭, open this folder as pacharm 클릭.
2. 위에 방법으로 인터프리터 설정
3. Tools ->Deployment -> configuration 가서   Mappings에서 Local path는 로컬, Deployment path는 보통 서버로 연결.
4.  다시 Tools-> Deployment에서 밑에 Automatic Load 체크되어있는지 확인
5. 만약 로컬과 remote의 경로가 다르면, 로컬에서 폴더를 생성하면 ㅍ파이참에서 보이지만, 리모트에서 생성하면 안보인다. 그대신, 연결은 되는데 이를 파이참에서 보이게 하려면, Tools-> Deployment -> Download하면 로컬로 내려오고, 파이참에서도 보인다.

6. 또한 말이 remote지 os.getcwd()하면 remote주소가 찍히므로 현재 주소는 remote이고 대산 쉬프트 스페이르를 눌렀을때 로컬의 폴더가 찍힐 뿐임.

 

**중요**
파이참 삭제는 아직 동기화 방법 못찾음
deployment는 모든 내가 설정한게 다 뜨므로 아무거라도 삭제하면 안됨
빈폴더는 로컬에서 만들어도 서버로 업데이트 안된다.(즉, 뭐라도 들어있어야됨)
좀 불편하긴 한데 보안때문에 쓴다고 생각하자.

 

위의 설정그대로 해도 안될때가 있다. 이게 컴퓨터 포맷하고 파이참 처음 설정하면서는 되는 설정인데 뭔진 모르겟지만 같은 서버에서 다른 가상환경 쓸때 안된다. 문제는 deployment 들어가면 왼쪽에 1개가 떠야 정상인데 여러개가 겹쳐서 뜨는 경우가 있음. 이럴때 configuration->Mappings에서 Deployment Path가 지정된거를 써야 되는데 이게 지정이 안되면 Errno2가 뜨면서 파일을 못찾는것임. 이게 그냥 new project만들어서 될때가 있고 안될때가 있음. 될때는 중복되는 이름이 없을때 된다. 여를 들어서 dash3에서 env1으로 하나 연결하고 env2로 하나 연결하면 env2는 안됨. 왜냐하면 deployment 들어가보면 dash3@115.145.190.164가 2개가 있어서 뒤에 (1)이 붙는 것을 볼 수 있다. 그래서 이걸 프로젝트 만들때마다 이름을 바꿔줘야 된다. 그러면 해결 OK. 또한 가끔씩 설정 잘못해서 자동으로 파일 업데이트하는게 안되는데 이건 default 서버 설정이 바껴서 그렇다. 그럴 땐 configuration에서 왼쪽 내가 이름바꾼 서버 오른쪽 클릭하면 use as a default 가 있다. 이걸 내가 원하는 서버로 설정하면 됨. (이전에 Dash3-KD_hanbeen 에서 작업하는데 Dash4로 되어 있어서 이러한 문제가 발생했다. Dash3에서는 Dash4를 접근할 수가 없어서 아예 연결이 안되었던 것이다.)

 

파이참에서 코드 작성하면 내가 만든 함수를 from import 할때가 있는데 이때 source root를 지정하면 안되는게 될 수 있다.

A
    A.py
    B.py

 

가 있을 때, A.py에서 B.py를 import 하려면 그냥 from B import funcB를 하면 된다. 그런데 파이참에서는 이게 빨간줄이 뜨면서 안되므로 디렉토리 A를 Source root로 해야한다. 그런데 그냥 콘솔 창에서 python A.py하면 되긴 한다. 즉, 파이참의 설정 문제이다. 