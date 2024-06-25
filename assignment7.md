# Assignment #7: April 月考

Updated 1557 GMT+8 Apr 3, 2024

2024 spring, Complied by石芯洁 数学科学学院 ==同学的姓名、院系==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c) 

Python编程环境：Spyder（Python 3.11)



## 1. 题目

### 27706: 逐词倒放

http://cs101.openjudge.cn/practice/27706/



思路：



代码

```python
# 
line=list(input().split( ))
ans=line[::-1]
print(*ans)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 081250.png)



### 27951: 机器翻译

http://cs101.openjudge.cn/practice/27951/



思路：



代码

```python
# 
def apps(line):
    ram=[]
    count=0
    while len(line)!=0:
        if line[0] in ram:
            line.pop(0)
        else:
            if len(ram)<M:
                ram.append(line.pop(0))
                count+=1
            else:
                ram.pop(0)
                ram.append(line.pop(0))
                count+=1
    return count
   
    
M,N=map(int,input().split())
line=list(map(int,input().split()))
print(apps(line))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 083802.png)



### 27932: Less or Equal

http://cs101.openjudge.cn/practice/27932/



思路：



代码

```python
# 
n,k=map(int,input().split())
data=list(map(int,input().split()))
data.sort()

if len(data)==k:
    print(data[-1])
    
elif k==0:
    if data[0]!=1:
        print(1)
    else:
        print(-1)
    
else:
    if data[k-1]==data[k]:
        print(-1)
    else:
        print(data[k-1])
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 090921.png)



### 27948: FBI树

http://cs101.openjudge.cn/practice/27948/



思路：



代码

```python
# 
def FBItree(data):
    if "0" in data and "1" in data:
        node="F"
    elif "1" in data:
        node="I"
    else:
        node="B"
    
    if len(data)>1:
        mid=len(data)//2
        left_tree=FBItree(data[:mid])
        right_tree=FBItree(data[mid:])
        return left_tree+right_tree+node
    else:
        return node

N=int(input())
data=input()
print(FBItree(data))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 093157.png)



### 27925: 小组队列

http://cs101.openjudge.cn/practice/27925/



思路：



代码

```python
# 
from collections import deque

t=int(input())
groups={}
member_to_group={}

for _ in range(t):
    members=list(map(int,input().split()))
    group_id=members[0]                     #把第一个队员的编号当队名
    groups[group_id]=deque()
    for member in members:
        member_to_group[member]=group_id    #每个队员的value是他们的队名

queue=deque()
queue_set=set()
    
while True:
    command=input().split()
    if command[0]=="STOP":
        break
    elif command[0]=="ENQUEUE":
        x=int(command[1])
        group=member_to_group.get(x,None)
        if group is None:    #散客
            group=x
            groups[group]=deque([x])
            member_to_group[x]=group
        else:
            groups[group].append(x) 
        if group not in queue_set:
            queue.append(group)
            queue_set.add(group)
    elif command[0]=="DEQUEUE":
        if queue:
            group=queue[0]
            x=groups[group].popleft()
            print(x)
            if not groups[group]:
                queue.popleft()
                queue_set.remove(group)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 161005.png)



### 27928: 遍历树

http://cs101.openjudge.cn/practice/27928/



思路：



代码

```python
# 
from collections import defaultdict
n = int(input())
tree = defaultdict(list)
parents = []
children = []

for i in range(n):
    t = list(map(int, input().split()))
    parents.append(t[0])
    if len(t) > 1:
        ch = t[1::]
        children.extend(ch)
        tree[t[0]].extend(ch)

def traversal(node):
    seq = sorted(tree[node] + [node])
    for x in seq:
        if x == node:
            print(node)
        else:
            traversal(x)

traversal((set(parents) - set(children)).pop())
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-09 162431.png)



## 2. 学习总结和收获

前两题可以较快做出来，第三题花了点时间寻找编码漏洞，后三题基本需要靠解题思路及题解学习。之前很少使用set，因此做题时几乎忽略了这个功能。通过这次作业，也对python中几个功能较熟悉了。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





