---
title:  "@staticmethod, @classmethod, self의 차이"
excerpt: "@staticmethod, @classmethod, self의 차이 기록"
categories:
  - python memo
  
tags:
  - python memo
  
toc: true
toc_sticky: true
toc_label: "On this page"
    
last_modified_at: 2021-06-29T10:48:00-05:00
---

### classmethod vs self
클래스메소드: 거의 self와 비슷한데, 아래 예제를 보자. 
![](/assets/images/2021-06-29-staticmethod_classmethod/user_func_def.JPG)
위의 코드와 같이 두개의 클래스를 만들었다. 하나는 classmethod, 하나는 instance method, 즉 self로 되어있다.
![](/assets/images/2021-06-29-staticmethod_classmethod/user_result.JPG) 
그 다음, 아래 코드처럼 불러오면, 클래스메소드는 cls자체가 포함된 클래스를 나타내는 변수이기 때문에 정상적으로 class를 __init__ 함수를 적용하여 return하는 것을 볼 수 있다.  
반면 self는 클래스를 리턴하지 못하고 이미 self자체로 하나의 인스턴스가 완성되었기 때문에 함수로 인지한다. 클래스 자체는 함수가 될 수 없기 때문에 callable에러가 뜨는 것을 볼 수 있다. 

### staticmethod
정적메소드는 하나의 클래스 안에서 이 정적메소드가 적용된 함수만 독립적으로 활동하게 할 때 쓴다. 즉, 외부로 영향을 안 주고 싶은것을 표현 할 때 사용한다. 따라서 아래와 같은 예제는 오류이다.
```python
class a:
    asd = "stat a"
    
    @staticmethod
    def func_a():
        print(asd)
```
func_a는 정적메소드이기 때문에 하나의 클래스안에 쓰여져 있더라도 클래스 변수를 참조 할 수 없다. 이걸 해결하려면 self.asd로 정의하거나, cls.asd 또는 a.asd로 해야한다. 
