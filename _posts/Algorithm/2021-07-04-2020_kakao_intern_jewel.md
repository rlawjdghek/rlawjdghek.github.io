---
title:  "(2020 카카오 인턴십) 보석 쇼핑"
excerpt: "(2020 카카오 인턴십) 보석 쇼핑"
categories:
  - Algorithm
  
tags:
  - Algorithm
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-07-04T14:48:00-05:00
---

### 문제 링크
[](https://programmers.co.kr/learn/courses/30/lessons/67258)

### 내 풀이
```python
from collections import defaultdict

def solution(gems):
    answer = [0,0]
    set_gems = set(gems)
    num_gems = len(set_gems)
    END = len(gems)
    num_check = defaultdict(lambda : 0)  # 주머니
    start = 0
    end = 0
    min_size = 100000  # 100000보다 클 수 없음

    # 다 안갖고 있으면 end가 늘어나면서 하나씩 넣고, 다 갖고있으면 start가 늘어나면서 하나씩 뺀다. 매 반복마다 다 갖고 있는경우 저장

    while True:  # end가 끝까지 갈 때까지 한다.
        if len(num_check) == num_gems:  # 만약 지금 다 갖고 있으면 앞에서부터 없애기
            num_check[gems[start]] -= 1
            if num_check[gems[start]] == 0:  # 만약 여분이 없으면 아예 주머니에서 빼기
                num_check.pop(gems[start])
            start += 1
        elif end == END:  # end가 끝을 지나면 종료 이 조건문이 2번째에 있어야 됨.
            break
        else:  # 이 조건은 다 갖고있지 않고 아직 end가 갈 길이 있을 때, 즉 end가 늘어나면서 주머니에 계속 넣어야된다.
            num_check[gems[end]] += 1  # default가 0이므로 항상 1부터 시작.
            end += 1
        if len(num_check) == num_gems: # 반복문 하나당 갯수를 계속 체크하면서 가장 작은 거리보다 작으면 업데이트 무조건 작아야함
            size = abs(end - start)
            if size < min_size:
                min_size = size
                answer[0] = start + 1
                answer[1] = end       
    return answer
```

### 주의할 점
주의할 점 : 처음 한 시도는 단순히 일단 뒤에서 부터 줄이면서 해당하는 보석을 줄이면서 0이 되면 그만 줄이고 그 다음부터 앞에서 같은 방법으로 줄였는데 이렇게 하면 반례가 존재한다. 

[4,1,2,5,2,2,2,1,2,3,4,5] 이면 뒤에 1,2,3,4,5 가 정답임에도 불구하고 5가 앞에 있어서 계속 줄인다. 
따라서 주머니를 텅 비운 상태로 뒤를 늘리면서 주머니에 모든 보석이 다 들어가면 저장해두고 앞에서 줄이면서 최솟값 찾고, 
다시 보석이 없으면 뒤를 늘려나가는 식으로 해야된다. 즉 , O(N)으로는 못 품. 또한 조건에서 명시한 것과 같이 앞에있는 것이 우선적이므로 앞에서부터 시
작하는 것을 기억. 즉 앞에서부터 시작하다가 뒤에서 같은 갯수 나오면 무시하는 방법으로 하면 된다. 


