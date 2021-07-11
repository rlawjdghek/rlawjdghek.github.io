---
title:  "(2020 카카오 인턴십) 키패드 누르기"
excerpt: "(2020 카카오 인턴십 키패드 누르기"
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
[](https://programmers.co.kr/learn/courses/30/lessons/67256)

그냥 하면 된다.

### 내 풀이
```python
def solution(numbers, hand):
    answer = ''
    LEFT_SET = {1,4,7}
    RIGHT_SET = {3,6,9}
    MIDDEL_SET = {2,5,8,10}
    left_pos = {"row" : 3, "col" : 0}
    right_pos = {"row" : 3, "col" : 2}
                  
    for number in numbers:
        print(number)
        if number == 0:
            number = 10
        if number in LEFT_SET:
            answer += "L"
            left_pos["row"] = number//3
            left_pos["col"] = 0
        elif number in RIGHT_SET:
            answer += "R"
            right_pos["row"] = number//3 - 1
            right_pos["col"] = 2
        else:
            number_row = number//3
            number_col = 1
            left_distance = abs(left_pos["row"] - number_row) + abs(left_pos["col"] - number_col)
            right_distance = abs(right_pos["row"] - number_row) + abs(right_pos["col"] - number_col)
           
            if left_distance < right_distance:
                ans = "L"
            elif left_distance > right_distance:
                ans = "R"
            else:
                ans = "L" if hand == "left" else "R"
                
            if ans == "L":
                left_pos["row"] = number_row
                left_pos["col"] = number_col
            else:
                right_pos["row"] = number_row
                right_pos["col"] = number_col
            answer += ans
     
    return answer
```
