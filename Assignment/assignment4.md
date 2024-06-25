# Assignment #4: 排序、栈、队列和树

Updated 0005 GMT+8 March 11, 2024

2024 spring, Complied by 石芯洁 数学科学学院 ==同学的姓名、院系==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:

Learn about Time complexities, learn the basics of individual Data Structures, learn the basics of Algorithms, and practice Problems.

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c)

Python编程环境：Spyder（Python 3.11)

## 1. 题目

### 05902: 双端队列

http://cs101.openjudge.cn/practice/05902/



思路：



代码

```python
# 
from collections import deque

def cod(n,lis):
    for _ in range(n):
        typ,num=map(int,input().split())
        if typ==1:
            lis.append(num)
        if typ==2:
            if num==1:
                lis.pop()
            else:
                lis.popleft()
    if lis:
        return True
    else:
        return False

t=int(input())
for i in range(t):
    n=int(input())
    lis=deque([])
    if cod(n,lis):
        print(*lis)
    else:
        print("NULL")
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-18 225510.png)



### 02694: 波兰表达式

http://cs101.openjudge.cn/practice/02694/



思路：



代码

```python
# 
sen=input().split()

def poland(sen):
    a=sen.pop(0)
    if a in "+ - * /":
        return str(eval(poland(sen) + a + poland(sen)))
    else:
        return a

print("%.6f" % float(poland(sen)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-18 225741.png)



### 24591: 中序表达式转后序表达式

http://cs101.openjudge.cn/practice/24591/



思路：



代码

```python
# 
def convert(n,st):
    let={"+":1,"-":1,"*":2,"/":2}
    stack=[]
    ans=[]
    num=""
    for char in st:
        if char.isnumeric() or char==".":
            num+=char
        else:
            if num:
                number=float(num)
                ans.append(int(number) if number.is_integer() else number)
                num=""
            if char in "+-*/":
                while stack and stack[-1] in "+-*/" and let[stack[-1]]>=let[char]:
                    ans.append(stack.pop())
                stack.append(char)
            elif char=="(":
                stack.append(char)
            elif char==")":
                while stack and stack[-1] in "+-*/":
                    ans.append(stack.pop())
                if stack[-1]=="(":
                    stack.pop()
    if num:
        number=float(num)
        ans.append(int(number) if number.is_integer() else number)
    while stack:
        ans.append(stack.pop())
    return " ".join(str(i) for i in ans)

n=int(input())
for _ in range(n):
    st=input()
    print(convert(n,st))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-19 091932.png)



### 22068: 合法出栈序列

http://cs101.openjudge.cn/practice/22068/



思路：



代码

```python
# 
def isPopSeq(s1,s2):
    stack=[]
    if len(s1)!=len(s2):
        return False
    else:
        L=len(s1)
        stack.append(s1[0])
        p1,p2=1,0
        while p1<L:
            if len(stack)>0 and stack[-1]==s2[p2]:
                stack.pop()
                p2+=1
            else:
                stack.append(s1[p1])
                p1+=1
        return "".join(stack[::-1])==s2[p2:]

s1=input()
while True:
    try:
        s2=input()
    except:
        break
    if isPopSeq(s1,s2):
        print("YES")
    else:
        print("NO")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-19 094930.png)



### 06646: 二叉树的深度

http://cs101.openjudge.cn/practice/06646/



思路：



代码

```python
# 
def max_depth(root):
    if root == -1:
        return 0
    else:
        left_depth = max_depth(tree[root][0])
        right_depth = max_depth(tree[root][1])
        return max(left_depth, right_depth) + 1

n = int(input())
tree = {}
for i in range(1, n + 1):
    left, right = map(int, input().split())
    tree[i] = (left, right)

print(max_depth(1))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-19 155343.png)



### 02299: Ultra-QuickSort

http://cs101.openjudge.cn/practice/02299/



思路：



代码

```python
# 
def merge_sort(arr):
    if len(arr) <= 1:
        return arr, 0
    
    mid = len(arr) // 2
    left, inv_left = merge_sort(arr[:mid])
    right, inv_right = merge_sort(arr[mid:])
    
    merged, inv_count = merge(left, right)
    total_inv = inv_left + inv_right + inv_count
    
    return merged, total_inv

def merge(left, right):
    merged = []
    inv_count = 0
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            inv_count += len(left) - i
            j += 1
    
    merged.extend(left[i:])
    merged.extend(right[j:])
    
    return merged, inv_count

while True:
    n = int(input())
    if n == 0:
        break
    sequence = [int(input()) for _ in range(n)]
    _, swaps = merge_sort(sequence)
    print(swaps)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-19 164255.png)



## 2. 学习总结和收获

题目有难度，单单是研究题目及理解答案逻辑就花了很长时间。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





