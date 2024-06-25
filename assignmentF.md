# Assignment #F: All-Killed 满分

Updated 1844 GMT+8 May 20, 2024

2024 spring, Complied by 石芯洁 数学科学学院 ==同学的姓名、院系==



**说明：**

1）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

2）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

3）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c) 

Python编程环境：Spyder（Python 3.11)



## 1. 题目

### 22485: 升空的焰火，从侧面看

http://cs101.openjudge.cn/practice/22485/



思路：



代码

```python
# 
def dfs(node,level):
    if ans[level]==0:
        ans[level]=node
    for nx in tree[node][::-1]:
        if nx !=-1:
            dfs(nx,level+1)

n=int(input())
tree={}
ans=[0]*n
for i in range(n):
    tree[i+1]=list(map(int,input().split()))
dfs(1,0)
see=[]
for i in ans:
    if i:
        see.append(i)

print(*see)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 084505.png)



### 28203:【模板】单调栈

http://cs101.openjudge.cn/practice/28203/



思路：



代码

```python
# 
n=int(input())
data=list(map(int,input().split()))
stack=[]

for i in range(n):
    for j in range(i,n):
        if data[j]>data[i]:
            stack.append(j+1)
            break
    else:
        stack.append(0)

print(*stack)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 091548.png)



### 09202: 舰队、海域出击！

http://cs101.openjudge.cn/practice/09202/



思路：



代码

```python
# 
from collections import defaultdict
def dfs(point):
    vis[point]=True
    for next_point in out[point]:
        inp[next_point]-=1
        if inp[next_point]==0:
            dfs(next_point)

t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    out=defaultdict(list)
    inp=[0]*(n+1)
    vis=[False]*(n+1)
    for i in range(m):
        x,y=map(int,input().split())
        out[x].append(y)
        inp[y]+=1
    for k in range(1,n+1):
        if inp[k]==0 and not vis[k]:
            dfs(k)
    flag=any(not vis[i] for i in range(1,n+1))
    print("Yes" if flag else "No")
        
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 094949.png)



### 04135: 月度开销

http://cs101.openjudge.cn/practice/04135/



思路：



代码

```python
# 
def check(cost):
    num,cut=1,0
    for i in range(n):
        if cut+spend[i]>cost:
            num+=1
            cut=spend[i]
        else:
            cut+=spend[i]
    if num>m:
        return False
    else:
        return True
    
n,m=map(int,input().split())
spend=[]
for _ in range(n):
    spend.append(int(input()))

minmax=max(spend)
maxmax=sum(spend)

while minmax < maxmax:
    middle=(minmax+maxmax)//2
    if check(middle):
        maxmax=middle
    else:
        minmax=middle+1

print(minmax)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 155608.png)



### 07735: 道路

http://cs101.openjudge.cn/practice/07735/



思路：



代码

```python
# 
import heapq
def dijkstra(graph):
    while path:
        dist,dest,fee=heapq.heappop(path)
        if dest==n-1:
            return dist
        for nex,leng,cost in graph[dest]:
            n_dist=dist+leng
            n_fee=fee+cost
            if n_fee<=k:
                dists[nex]=n_dist
                heapq.heappush(path,(n_dist,nex,n_fee))
    return -1

k=int(input())
n=int(input())
r=int(input())
graph=[[] for _ in range(n)]
for _ in range(r):
    s,d,l,t=map(int,input().split())
    graph[s-1].append((d-1,l,t))
    
path=[(0,0,0)]
dists=[float("inf")]*n
dists[0]=0
result=dijkstra(graph)
print(result)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 172639.png)



### 01182: 食物链

http://cs101.openjudge.cn/practice/01182/



思路：



代码

```python
# 
def find(x):	
    if p[x] == x:
        return x
    else:
        p[x] = find(p[x])	
        return p[x]

n,k = map(int, input().split())

p = [0]*(3*n + 1)
for i in range(3*n+1):	
    p[i] = i

ans = 0
for _ in range(k):
    a,x,y = map(int, input().split())
    if x>n or y>n:
        ans += 1; continue
    
    if a==1:
        if find(x+n)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        
        p[find(x)] = find(y)				
        p[find(x+n)] = find(y+n)
        p[find(x+2*n)] = find(y+2*n)
    else:
        if find(x)==find(y) or find(y+n)==find(x):
            ans += 1; continue
        p[find(x+n)] = find(y)
        p[find(y+2*n)] = find(x)
        p[find(x+2*n)] = find(y+n)

print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-28 173840.png)



## 2. 学习总结和收获

题目难，几乎无法独立完成，需通过题解学习。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





