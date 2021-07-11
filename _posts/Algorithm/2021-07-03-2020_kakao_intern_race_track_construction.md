---
title:  "(2020 카카오 인턴십) 경주로 건설"
excerpt: "(2020 카카오 인턴십) 경주로 건설"
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
[](https://programmers.co.kr/learn/courses/30/lessons/67259)

### 내 풀이
```python
def solution(board):
    answer = 0
    DX = [1,0,-1,0]
    DY = [0,1,0,-1]
    DIR = [0,1,2,3]  # x축은 0, y축은 1
    N_ROW = len(board)
    N_COL = N_ROW

    queue = [(0,0)]  # (row, col, 지금까지 오면서 베스트 경로의 현재 방향(1이면 x축 -1이면 y축))
    cost = [[[99999999 for _ in range(4)] for _ in range(N_COL)] for _ in range(N_ROW)]  # 4개의 방향도 저장해두어야 한다.
    # 맨 처음에는 방향을 넣기 싫어서 한번은 그냥 돌린다. 또한 모든 방향 고려 안하고 동쪽, 남쪽만
    cur_x, cur_y = queue.pop(0)
    for i, (dir_x, dir_y) in enumerate(zip(DX[:2], DY[:2])):
        next_x = dir_x
        next_y = dir_y
       
        if not board[next_x][next_y]:
            queue.append((next_x, next_y, i, 100))
            cost[next_x][next_y][i] = 100

    while len(queue) != 0:
        cur_x, cur_y, cur_dir, cur_cost = queue.pop(0)
        for dx, dy, dir_ in zip(DX, DY, DIR):
            next_x = cur_x + dx
            next_y = cur_y + dy
            if next_x > -1 and next_y > -1 and next_x < N_ROW and next_y < N_COL:  # map 밖으로 넘어가면 안됨.
                if not board[next_x][next_y]:  # 벽이 아니면
                    if cur_dir == dir_:  # 지금 있는 방향과 다음에 갈 방향이 같으면 직선만
                        if cost[next_x][next_y][dir_] > cur_cost + 100:
                            cost[next_x][next_y][dir_] = cur_cost + 100
                            queue.append((next_x, next_y, dir_, cur_cost + 100))
                    else:
                        if cost[next_x][next_y][dir_] > cur_cost + 600:
                            cost[next_x][next_y][dir_] = cur_cost + 600
                            queue.append((next_x, next_y, dir_, cur_cost + 600))
    print(cost)
    answer = min(cost[N_ROW-1][N_COL-1])
    return answer
```

### 주의할 점
cost 배열을 3차원으로 안하고 2차원으로만 하면 늦게 도착하는데 나중에 있는 커브 때문에 결과적으로 더 싼 경우가 존재한다. 아래 그림 참고
![](/assets/images/2021-07-03-2020_kakao_intern_race_track_construction/7.JPG)