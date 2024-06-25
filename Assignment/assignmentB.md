# Assignment #B: 图论和树算

Updated 1709 GMT+8 Apr 28, 2024

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

### 28170: 算鹰

dfs, http://cs101.openjudge.cn/practice/28170/



思路：



代码

```python
# 
def dfs(x,y):
    graph[x][y]="%"
    for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
        if 0<=x+dx<10 and 0<=y+dy<10 and graph[x+dx][y+dy]==".":
            dfs(x+dx,y+dy)

graph=[]
result=0
for i in range(10):
    graph.append(list(input()))

for i in range(10):
    for j in range(10):
        if graph[i][j]==".":
            result+=1
            dfs(i,j)

print(result)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 082247.png)



### 02754: 八皇后

dfs, http://cs101.openjudge.cn/practice/02754/



思路：



代码

```python
# 
a=[]
def q(l,n):
    for i in "12345678":
        if not(i in n or sum(abs(int(i) - int(n[j])) == l - j for j in range(l))):
            if l > 6:
                a.append(n + i)
            else:
                q(l + 1, n + i)
q(0, "")
for _ in range(int(input())):
    print(a[int(input()) - 1])
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 083321.png)



### 03151: Pots

bfs, http://cs101.openjudge.cn/practice/03151/



思路：



代码

```python
# 
def bfs(A, B, C):
    start = (0, 0)
    visited = set()
    visited.add(start)
    queue = [(start, [])]

    while queue:
        (a, b), actions = queue.pop(0)

        if a == C or b == C:
            return actions

        next_states = [(A, b), (a, B), (0, b), (a, 0), (min(a + b, A),\
                max(0, a + b - A)), (max(0, a + b - B), min(a + b, B))]

        for i in next_states:
            if i not in visited:
                visited.add(i)
                new_actions = actions + [get_action(a, b, i)]
                queue.append((i, new_actions))

    return ["impossible"]


def get_action(a, b, next_state):
    if next_state == (A, b):
        return "FILL(1)"
    elif next_state == (a, B):
        return "FILL(2)"
    elif next_state == (0, b):
        return "DROP(1)"
    elif next_state == (a, 0):
        return "DROP(2)"
    elif next_state == (min(a + b, A), max(0, a + b - A)):
        return "POUR(2,1)"
    else:
        return "POUR(1,2)"


A, B, C = map(int, input().split())
solution = bfs(A, B, C)

if solution == ["impossible"]:
    print(solution[0])
else:
    print(len(solution))
    for i in solution:
        print(i)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 090457.png)



### 05907: 二叉树的操作

http://cs101.openjudge.cn/practice/05907/



思路：



代码

```python
# 
class treenode:
    def __init__(self,value=0):
        self.left=None
        self.right=None
        self.value=value

def build(nodes_info):
    nodes=[treenode(i) for i in range(n)]
    for value,left,right in nodes_info:
        if left !=-1:
            nodes[value].left=nodes[left]
        if right !=-1:
            nodes[value].right=nodes[right]
    return nodes

def type1(nodes,x,y):
    for node in nodes:
        if node.left and node.left.value in[x,y]:
            node.left=nodes[y] if node.left.value==x else nodes[x]
        if node.right and node.right.value in [x,y]:
            node.right=nodes[y] if node.right.value==x else nodes[x]

def type2(node):
    while node and node.left:
        node=node.left
    return node.value if node else -1
    
t=int(input())
for _ in range(t):
    n,m=map(int,input().split())
    nodes_info=[tuple(map(int,input().split())) for _ in range(n)]
    ops=[tuple(map(int,input().split())) for _ in range(m)]
    nodes=build(nodes_info)
    for op in ops:
       if op[0]==1:
           type1(nodes,op[1],op[2])
       elif op[0]==2:
           print(type2(nodes[op[1]]))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 094809.png)





### 18250: 冰阔落 I

Disjoint set, http://cs101.openjudge.cn/practice/18250/



思路：



代码

```python
# 
def find(x):
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(x, y):
    root_x = find(x)
    root_y = find(y)
    if root_x != root_y:
        parent[root_y] = root_x

while True:
    try:
        n, m = map(int, input().split())
        parent = list(range(n + 1))

        for _ in range(m):
            a, b = map(int, input().split())
            if find(a) == find(b):
                print('Yes')
            else:
                print('No')
                union(a, b)

        unique_parents = set(find(x) for x in range(1, n + 1))  
        ans = sorted(unique_parents) 
        print(len(ans))
        print(*ans)

    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 154610.png)



### 05443: 兔子与樱花

http://cs101.openjudge.cn/practice/05443/



思路：



代码

```python
# 
import heapq
import math
def dijkstra(graph,start,end,P):
    if start == end: return []
    dist = {i:(math.inf,[]) for i in graph}
    dist[start] = (0,[start])
    pos = []
    heapq.heappush(pos,(0,start,[]))
    while pos:
        dist1,current,path = heapq.heappop(pos)
        for (next,dist2) in graph[current].items():
            if dist2+dist1 < dist[next][0]:
                dist[next] = (dist2+dist1,path+[next])
                heapq.heappush(pos,(dist1+dist2,next,path+[next]))
    return dist[end][1]

P = int(input())
graph = {input():{} for _ in range(P)}
for _ in range(int(input())):
    place1,place2,dist = input().split()
    graph[place1][place2] = graph[place2][place1] = int(dist)

for _ in range(int(input())):
    start,end = input().split()
    path = dijkstra(graph,start,end,P)
    s = start
    current = start
    for i in path:
        s += f'->({graph[current][i]})->{i}'
        current = i
    print(s)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-05-07 175411.png)



## 2. 学习总结和收获

这次题目对我来说还是有难度，但大概看了题解后基本能够了解并写出代码。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





