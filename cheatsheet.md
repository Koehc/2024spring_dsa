## 各种模板+复习题目

#### 01258: Agri-Net（最小生成树模板）

Farmer John has been elected mayor of his town! One of his campaign promises was to bring internet connectivity to all farms in the area. He needs your help, of course. Farmer John ordered a high speed connection for his farm and is going to share his connectivity with the other farmers. To minimize cost, he wants to lay the minimum amount of optical fiber to connect his farm to all the other farms. Given a list of how much fiber it takes to connect each pair of farms, you must find the minimum amount of fiber needed to connect them all together. Each farm must connect to some other farm such that a packet can flow from any one farm to any other farm. The distance between any two farms will not exceed 100,000.

**输入**

The input includes several cases. For each case, the first line contains the number of farms, N (3 <= N <= 100). The following lines contain the N x N conectivity matrix, where each element shows the distance from on farm to another. Logically, they are N lines of N space-separated integers. Physically, they are limited in length to 80 characters, so some lines continue onto others. Of course, the diagonal will be 0, since the distance from farm i to itself is not interesting for this problem.

**输出**

For each case, output a single integer length that is the sum of the minimum length of fiber required to connect the entire set of farms.

样例输入

```
4
0 4 9 21
4 0 8 17
9 8 0 16
21 17 16 0
```

样例输出

```
28
```

```python
def P(x):
    if p[x] != x:
        p[x] = P(p[x])
    return p[x]
while True:
    try:
        n = int(input())
    except EOFError:
        break
    ans = 0
    M = [list(map(int, input().split())) for _ in range(n)]
    p = [i for i in range(n)]
    l = []
    for i in range(n):
        for j in range(n):
            if i != j:
                l.append((i, j, M[i][j]))
    l.sort(key=lambda x: x[2])
    for i, j, k in l:
        pi, pj = P(i), P(j)
        if pi != pj:
            p[pi] = pj
            ans += k
    print(ans)
```



#### 最短路径（dijkstra)

```python
import heapq
def dijkstra(graph, start, end):
    distances = {vertex: float('inf') for vertex in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        if current_vertex == end:
            path = []
            while current_vertex != start:
                path.append(current_vertex)
                current_vertex = previous_vertices[current_vertex]
            path.append(start)
            path.reverse()
            return path, current_distance
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_vertices[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
    return None
```



#### **常用结构与方法**

##### **1.并查集:**

列表实现

```python
def find(x):
    if p[x]!=x:
        p[x]=find(p[x])
    return p[x]

def union(x,y):
    rootx,rooty=find(x),find(y)
    if rootx!=rooty:
        p[rootx]=p[rooty]

#单纯并查
p=list(range(n+1))  
unique_parents = set(find(x) for x in range(1, n + 1))  最后收取数据

#反向事件  用x+n储存x的反向事件，查询时直接find（x+n）
p=list(range(2*(n+1))
if tag=="Different":
	union(x,y+n)
    union(y,x+n)
```

集合实现：

```py
class UnionFind:
    def __init__(self,n):
        self.p=list(range(n))
        self.h=[0]*n
    def find(self,x):
        if self.p[x]!=x:
            self.p[x]=self.find(self.p[x])
        return self.p[x]
    def union(self,x,y):
        rootx=self.find(x)
        rooty=self.find(y)
        if rootx!=rooty:
            if self.h[rootx]<self.h[rooty]:
                self.p[rootx]=rooty
            elif self.h[rootx]>self.h[rooty]:
                self.p[rooty]=rootx
            else:
                self.p[rooty]=rootx
                self.h[rootx]+=1
```

---

##### **2.堆：**

```python
from heapq import *

heappush(heap, item)
heappop(heap) 弹出最小，可接收
heap[0] 访问最小，可接收
heapify(lst) 建堆
heapreplace(heap, item)
heappushpop(heap, item)
```

默认为最小堆，转化为最大堆方法：全部化为负数，输出时再次加负号

##### **3.双端队列：**

```py
from collections import deque
q=deque()
q.appendleft()
q.append()
q.popleft()
q.pop()
```

##### **4.二分查找：**

```py
import bisect
sorted_list = [1,3,5,7,9] 
position = bisect.bisect_left(sorted_list, 6) #查找元素应左插入的位置
print(position)  # 输出：3，因为6应该插入到位置3，才能保持列表的升序顺序

bisect.insort_left(sorted_list, 6) #左插入元素
print(sorted_list)  # 输出：[1, 3, 5, 6, 7, 9]，6被插入到适当的位置以保持升序顺序

sorted_list=(1,3,5,7,7,7,9)
print(bisect.bisect_left(sorted_list,7))
print(bisect.bisect_right(sorted_list,7))
```

##### **5.扩栈：**

```py
import sys
sys.setrecursionlimit(1<<30)
```

##### **Stack模板：**

##### **逆波兰表达式求值：**

```python
stack=[]
for t in s:
    if t in '+-*/':
        b,a=stack.pop(),stack.pop()
        stack.append(str(eval(a+t+b)))
    else:
        stack.append(t)
print(f'{float(stack[0]):.6f}')
```

---

##### 最大全0子矩阵:

```python
for row in ma:
    stack=[]
    for i in range(n):
        h[i]=h[i]+1 if row[i]==0 else 0
        while stack and h[stack[-1]]>h[i]:
            y=h[stack.pop()]
            w=i if not stack else i-stack[-1]-1
            ans=max(ans,y*w)
        stack.append(i)
    while stack:
        y=h[stack.pop()]
        w=n if not stack else n-stack[-1]-1
        ans=max(ans,y*w)
print(ans)
```

---

##### 求逆序对数:

```python
from bisect import *
a=[]
rev=0
for _ in range(n):
    num=int(input())
    rev+=bisect_left(a,num)
    insort_left(a,num)
ans=n*(n-1)//2-rev
```

```python
def merge_sort(a):
    if len(a)<=1:
        return a,0
    mid=len(a)//2
    l,l_cnt=merge_sort(a[:mid])
    r,r_cnt=merge_sort(a[mid:])
    merged,merge_cnt=merge(l,r)
    return merged,l_cnt+r_cnt+merge_cnt
def merge(l,r):
    merged=[]
    l_idx,r_idx=0,0
    inverse_cnt=0
    while l_idx<len(l) and r_idx<len(r):
        if l[l_idx]<=r[r_idx]:
            merged.append(l[l_idx])
            l_idx+=1
        else:
            merged.append(r[r_idx])
            r_idx+=1
            inverse_cnt+=len(l)-l_idx
    merged.extend(l[l_idx:])
    merged.extend(r[r_idx:])
    return merged,inverse_cnt
```



#### **Tree模板**

##### **1.二叉树深度：**

```python
def tree_depth(node):
    if node is None:
        return 0
    left_depth = tree_depth(node.left)
    right_depth = tree_depth(node.right)
    return max(left_depth, right_depth) + 1
```

---

##### **2.二叉树的读取与建立：**

输入为每个节点的子节点：

```python
nodes = [TreeNode() for _ in range(n)]

for i in range(n):
    left_index, right_index = map(int, input().split())
    if left_index != -1:
        nodes[i].left = nodes[left_index]
    if right_index != -1:
        nodes[i].right = nodes[right_index]
```

但要注意，这里的index随题目要求而改变，即从0开始还是从1开始的问题，可能要-1

---

括号嵌套树的解析建立：

```python
def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():  # 如果是字母，创建新节点
            node = TreeNode(char)
            if stack:  # 如果栈不为空，把节点作为子节点加入到栈顶节点的子节点列表中
                stack[-1].children.append(node)
        elif char == '(':  # 遇到左括号，当前节点可能会有子节点
            if node:
                stack.append(node)  # 把当前节点推入栈中
                node = None
        elif char == ')':  # 遇到右括号，子节点列表结束
            if stack:
                node = stack.pop()  # 弹出当前节点
    return node  # 根节点
```

---

根据前中序得后序，根据中后序得前序：

```python
def postorder(preorder,inorder):
    if not preorder:
        return ''
    root=preorder[0]
    idx=inorder.index(root)
    left=postorder(preorder[1:idx+1],inorder[:idx])
    right=postorder(preorder[idx+1:],inorder[idx+1:])
    return left+right+root
```

```python
def preorder(inorder,postorder):
    if not inorder:
        return ''
    root=postorder[-1]
    idx=inorder.index(root)
    left=preorder(inorder[:idx],postorder[:idx])
    right=preorder(inorder[idx+1:],postorder[idx:-1])
    return root+left+right
```

---

##### **3.二叉树叶节点计数**：

```python
def count_leaves(node):
    if node is None:
        return 0
    if node.left is None and node.right is None:
        return 1
    return count_leaves(node.left)+count_leaves(node.right)
```

---

##### **4.树的遍历：**

前/后序遍历：

```python
def preorder(node):
    output = [node.value]
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def preorder(node):
    if node is not None:
        return tree.value+preorder(tree.left)+preorder(tree.right)
    else:
        return ""

def postorder(node):
    output = []
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

def postorder(node):
    if node is not None:
        return postorder(node.left)+postorder(node.right)+node.value
    else:
        return ""
```

---

中序遍历：

```python
def inorder(tree):
    if tree is not None:
        return inorder(tree.left)+tree.value+inorder(tree.right)
    else:
        return ""
```

---

分层遍历：（使用bfs）

```python
from collections import deque

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

def level_Order(root):
    queue = deque()
    queue.append(root)
    
    while len(queue) != 0: # 这里是一个特殊的BFS,以层为单位
        n = len(queue)     
        while n > 0: #一层层的输出结果
            point = queue.popleft()
            print(point.value, end=" ") # 这里的输出是一行
            queue.extend(point.children)
            n -= 1
            
        print()   #要加上 end的特殊语法
```

---

```py
from collections import deque
def levelorder(root):
    if not root:
        return ""
    q=deque([root])  
    res=""
    while q:
        node=q.popleft()  
        res+=node.val  
        if node.left:
            q.append(node.left)
        if node.right:
            q.append(node.right)
    return res
```



**例：** **#### 22485: 升空的焰火，从侧面看（树，bfs)**

http://cs101.openjudge.cn/practice/22485/

生态文明建设是关系中华民族永续发展的根本大计。近年来，为了响应环保号召，商家们研制出了环保烟花。这类烟花在烟花配方中不采用含有重金属和硫元素的物质，从而减少了硫化物的生成。

为了庆祝院庆，A大学计算机院燃放了一批环保烟花。从正面看，烟花的构成了二叉树的形状。那么从侧面看，烟花又是什么样子的呢？

对于一个二叉树形状的烟花，它有N个节点，每个节点都有一个1~N之间的颜色编号，不同节点的编号互不相同。除了根节点的编号固定为1，其他节点的编号都是随机分配的。

我们需要按照从顶部到底部的顺序，输出从右侧能看到的节点的颜色编号，即**输出广度优先搜索中每一层最后一个节点**。

例如对于如下的二叉树烟花，从右侧看到的结果为[1, 3, 4]。

[![img](https://camo.githubusercontent.com/8e4536b02e595816886d132b30eed0c0a8552302bb4851955658d13a4548b461/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f353136372f313632323033353130362e706e67)](https://camo.githubusercontent.com/8e4536b02e595816886d132b30eed0c0a8552302bb4851955658d13a4548b461/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f353136372f313632323033353130362e706e67)

再如，对于如下的二叉树烟花，从右侧看到的结果为[1, 7, 5, 6, 2]。

[![img](https://camo.githubusercontent.com/f511d5401105522c5448a551541968f975ce085269f101ae28af5d1ea6ce715f/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f363430382f313632323732383539362e706e67)](https://camo.githubusercontent.com/f511d5401105522c5448a551541968f975ce085269f101ae28af5d1ea6ce715f/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f363430382f313632323732383539362e706e67)

**输入**

输入共N+1行。 第1行为一个整数N（1<=N<=1000），表示二叉树中的节点个数。这N个节点的颜色编号分别为1到N，其中1号节点为根节点。 接下来N行每行有两个整数，分别为1~N号节点的左子节点和右子节点的颜色编号，如果子节点为空，则用-1表示。

**输出**

按从顶到底的顺序，输出从右侧看二叉树看到的各个节点的颜色编号（即广度优先搜索中每一层最后一个节点），每个编号之间用空格隔开。

样例输入

```
5
2 3
-1 5
-1 4
-1 -1
-1 -1
```



样例输出

```
1 3 4
```

提示

（1）一种处理本题的输入形式的方式：可先开辟一个大小为N的数组，存储这N个二叉树节点，然后根据每行输入，将相关节点用左右子节点指针连接起来。 （2）BFS可以借助队列实现，可以使用STL

```python
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



##### **5.二叉搜索树的构建：**

```py
def insert(root,num):
    if not root:
        return Node(num)
    if num<root.val:
        root.left=insert(root.left,num)
    else:
        root.right=insert(root.right,num)
    return root
```

---

##### **6.字典树的构建：**

```py
def insert(root,num):
    node=root
    for digit in num:
        if digit not in node.children:
            node.children[digit]=TrieNode()
        node=node.children[digit]
        node.cnt+=1
```



#### **Graph模版**

##### **1.Dijikstra：**

```py
#用字典储存路径
ways=dict()
for _ in range(p):
    ways[input()]=[]
q=int(input())
for i in range(q):
    FRM,TO,CST=input().split()
    ways[FRM].append((TO,int(CST)))
    ways[TO].append((FRM,int(CST)))

#函数主体(带路径的实现)
from heapq import *
def dijkstra(frm,to):
    q=[]
    tim=0
    heappush(q,(tim,frm,[frm]))
    visited=set([frm])
    if frm==to:
        return frm,0
    while q:
        tim,x,how=heappop(q)
        if x==to:
            return "->".join(how),tim
        visited.add(x)
        for way in ways[x]:
            nx=way[0];cst=way[1]
            if nx not in visited:
                nhow=how.copy()
                nhow.append(f"({cst})")
                nhow.append(nx)
                heappush(q,(tim+cst,nx,nhow))
    return "NO"
```

注意visited是为了在无向图中防止返回，有向图不需要visited

---

##### **2.BFS：**

```py
def bfs(graph, initial):
    visited = set()
    queue = [(initial,tim)]
 
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.add(node)
            neighbours = graph[node]
            nt=tim
 
            for neighbour in neighbours:
                cst=costs[neighbour]
                queue.append((neighbour,cst+tim)
```

---

##### **3.DFS：**

```py
def dfs(v):
    visited.add(v)
    total = values[v]  #以最大权值联通块为例
    for w in graph[v]:
        if w not in visited:
            total += dfs(w)
    return total
```

------

##### **4.Prim：**

用途：在N**2时间内实现最小生成树。

```py
from heapq import *
def prim(graph, n):
    vis = [False] * n
    mh = [(0, 0)]  # (weight, vertex)
    mc = 0
    while mh:
        wei, ver = heappop(mh)
    	if vis[vertex]:
            continue
        vis[ver] = True
        mc += wei
        for nei, nw in graph[ver]:
            if not vis[nei]:
                heappush(mh, (nw, nei))
    return mc if all(visited) else -1

def main():
    n, m = map(int, input().split())
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u, v, w = map(int, input().split())
        graph[u].append((v, w))
        graph[v].append((u, w))
    mc = prim(graph, n)
    print(mc)

if __name__ == "__main__":
    main()
```

------



### 一些题目...

#### 09202: 舰队、海域出击！（拓扑排序检查有向图环）

**描述**

作为一名海军提督，Pachi将指挥一支舰队向既定海域出击！ Pachi已经得到了海域的地图，地图上标识了一些既定目标和它们之间的一些单向航线。如果我们把既定目标看作点、航线看作边，那么海域就是一张有向图。不幸的是，Pachi是一个会迷路的提督QAQ，所以他在包含环(圈)的海域中必须小心谨慎，而在无环的海域中则可以大展身手。 受限于战时的消息传递方式，海域的地图只能以若干整数构成的数据的形式给出。作为舰队的通讯员，在出击之前，请你告诉提督海域中是否包含环。

例如下面这个海域就是无环的：

[![img](https://camo.githubusercontent.com/0c72eb1559213ae2e7d198549857c2ba77ec29a1670504456a578e3d3002b0e3/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f313435303638363438322e706e67)](https://camo.githubusercontent.com/0c72eb1559213ae2e7d198549857c2ba77ec29a1670504456a578e3d3002b0e3/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f313435303638363438322e706e67)

而下面这个海域则是有环的（C-E-G-D-C）：

[![img](https://camo.githubusercontent.com/d6298148d53d04e0a87355dbf687935ca162791c254df09f940bc6cae647fef1/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f313435303638363534342e706e67)](https://camo.githubusercontent.com/d6298148d53d04e0a87355dbf687935ca162791c254df09f940bc6cae647fef1/687474703a2f2f6d656469612e6f70656e6a756467652e636e2f696d616765732f75706c6f61642f313435303638363534342e706e67)

**输入**

每个测试点包含多组数据，每组数据代表一片海域，各组数据之间无关。 第一行是数据组数T。 每组数据的第一行两个整数N，M，表示海域中既定目标数、航线数。 接下来M行每行2个不相等的整数x,y，表示从既定目标x到y有一条单向航线（所有既定目标使用1~N的整数表示）。 描述中的图片仅供参考，其顶点标记方式与本题数据无关。

1<=N<=100000，1<=M<=500000，1<=T<=5 注意：输入的有向图不一定是连通的。

**输出**

输出包含T行。 对于每组数据，输出Yes表示海域有环，输出No表示无环。

**样例输入**

```
2
7 6
1 2
1 3
2 4
2 5
3 6
3 7
12 13
1 2
2 3
2 4
3 5
5 6
4 6
6 7
7 8
8 4
7 9
9 10
10 11
10 12
```

**样例输出**

```
No
Yes
```

**提示**

输入中的两张图就是描述中给出的示例图片。

拓扑排序检查有向图是否存在环



```python
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



#### **04135: 月度开销（二分查找+贪心）**

**描述**

农夫约翰是一个精明的会计师。他意识到自己可能没有足够的钱来维持农场的运转了。他计算出并记录下了接下来 *N* (1 ≤ *N* ≤ 100,000) 天里每天需要的开销。

约翰打算为连续的*M* (1 ≤ *M* ≤ *N*) 个财政周期创建预算案，他把一个财政周期命名为fajo月。每个fajo月包含一天或连续的多天，每天被恰好包含在一个fajo月里。

约翰的目标是合理安排每个fajo月包含的天数，使得开销最多的fajo月的开销尽可能少。

**输入**

第一行包含两个整数N,M，用单个空格隔开。
接下来N行，每行包含一个1到10000之间的整数，按顺序给出接下来N天里每天的开销。

**输出**

一个整数，即最大月度开销的最小值。

**样例输入**

```
7 5
100
400
300
100
500
101
400
```

**样例输出**

```
500
```

**提示**

若约翰将前两天作为一个月，第三、四两天作为一个月，最后三天每天作为一个月，则最大月度开销为500。其他任何分配方案都会比这个值更大。

```python
def check(cost):
    num,cut=1,0                     #num=fajo数，cut=当前分割内的开销
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
    #二分法，每次看可达成数额在少的一半还是多的一半，直到确定值出现
    if check(middle):              
        maxmax=middle              #可达成mean最大值在小的一半所以maxmax变小
    else:
        minmax=middle+1            #不可达成so minmax变大
print(minmax)
```



#### **02774: 木材加工(二分查找)**

**描述**

木材厂有一些原木，现在想把这些木头切割成一些长度相同的小段木头，需要得到的小段的数目是给定了。当然，我们希望得到的小段越长越好，你的任务是计算能够得到的小段木头的最大长度。

木头长度的单位是厘米。原木的长度都是正整数，我们要求切割得到的小段木头的长度也要求是正整数。

**输入**

第一行是两个正整数*N*和*K*(1 ≤ *N* ≤ 10000, 1 ≤ *K* ≤ 10000)，*N*是原木的数目，*K*是需要得到的小段的数目。 接下来的*N*行，每行有一个1到10000之间的正整数，表示一根原木的长度。 　

**输出**

输出能够切割得到的小段的最大长度。如果连1厘米长的小段都切不出来，输出"0"。

**样例输入**

```
3 7
232
124
456
```

**样例输出**

```
114
```

```python
def check(x):
    count=0
    for i in range(n):
        count+=woods[i]//x
    return count >=k
n,k=map(int,input().split())
woods=[]
for _ in range(n):
    woods.append(int(input()))
maxmax=max(woods)+1
minmax=1                         
if sum(woods)<k:
    print(0)
    exit()
while maxmax > minmax:
    middle=(maxmax+minmax)//2
    if check(middle):
        ans=middle
        minmax=middle+1
    else:
        maxmax=middle
print(ans)
#二分查找终止条件为 min=max! 
#min=答案达成条件的最小可能性，max=最大可能性 
#要有middle
```



#### 07735: 道路(dijkstra+dfs)

**描述**

N个以 1 ... N 标号的城市通过单向的道路相连:。每条道路包含两个参数：道路的长度和需要为该路付的通行费（以金币的数目来表示）

Bob and Alice 过去住在城市 1.在注意到Alice在他们过去喜欢玩的纸牌游戏中作弊后，Bob和她分手了，并且决定搬到城市N。他希望能够尽可能快的到那，但是他囊中羞涩。我们希望能够帮助Bob找到从1到N最短的路径，前提是他能够付的起通行费。

**输入**

第一行包含一个整数K, 0 <= K <= 10000, 代表Bob能够在他路上花费的最大的金币数。第二行包含整数N， 2 <= N <= 100, 指城市的数目。第三行包含整数R, 1 <= R <= 10000, 指路的数目.
接下来的R行，每行具体指定几个整数S, D, L 和 T来说明关于道路的一些情况，这些整数之间通过空格间隔:
S is 道路起始城市, 1 <= S <= N
D is 道路终点城市, 1 <= D <= N
L is 道路长度, 1 <= L <= 100
T is 通行费 (以金币数量形式度量), 0 <= T <=100
注意不同的道路可能有相同的起点和终点。

**输出**

输入结果应该只包括一行，即从城市1到城市N所需要的最小的路径长度（花费不能超过K个金币）。如果这样的路径不存在，结果应该输出-1。

**样例输入**

```
5
6
7
1 2 2 3
2 4 3 3
3 4 2 4
1 3 4 1
4 6 2 1
3 5 2 0
5 4 3 2
```

**样例输出**

```
11
```

```python
#dijkstra变种
import heapq
def dijkstra(graph):
    while path:
        dist,dest,fee=heapq.heappop(path)        # A to B #heappop确保从最短距离的point开始搜索
        if dest==n-1:
            return dist                          #最短路径以累计到终点
        for nex,leng,cost in graph[dest]:        # B to C
            n_dist=dist+leng                     # A to C
            n_fee=fee+cost
            if n_fee<=k:
                dists[nex]=n_dist
                heapq.heappush(path,(n_dist,nex,n_fee))    
                #增加新的累积后的可能路线
    return -1
k=int(input())
n=int(input())
r=int(input())
graph=[[] for _ in range(n)]             #建一个邻接表储存每个点与其他点的关系
for _ in range(r):
    s,d,l,t=map(int,input().split())     
    #s=start d=destination l=distance t=fee
    graph[s-1].append((d-1,l,t))    
path=[(0,0,0)]                           
#去到初始城市的数据（距离，城市，花费）
dists=[float("inf")]*n                   #储存去到n个城市的距离
dists[0]=0
result=dijkstra(graph)
print(result)
```



#### 28203:【模板】单调栈

**描述**

给出项数为 n 的整数数列 a1...an。

定义函数 f(i) 代表数列中第 i 个元素之后第一个大于 ai 的元素的**下标**，。若不存在，则 f(i)=0。试求出 f(1...n)。

**输入**

第一行一个正整数 n。 第二行 n 个正整数 a1...an。

**输出**

一行 n 个整数表示 f(1), f(2), ..., f(n) 的值。

**样例输入**

```
5
1 4 2 3 5
```

**样例输出**

```
2 5 4 5 0
```

```python
n = int(input())
a = list(map(int, input().split()))
stack = []

#f = [0]*n
for i in range(n):
    while stack and a[stack[-1]] < a[i]:
        #f[stack.pop()] = i + 1
        a[stack.pop()] = i + 1


    stack.append(i)

while stack:
    a[stack[-1]] = 0
    stack.pop()

print(*a)
```



#### 04099: 队列和栈

**描述**

现在，假设队列和栈都是空的。给定一系列push k和pop操作之后，输出队列和栈中存的数字。若队列或栈已经空了，仍然接收到pop操作，则输出error。

**输入**

第一行为m，表示有m组测试输入，m<100。 每组第一行为n，表示下列有n行push k或pop操作。（n<150） 接下来n行，每行是push k或者pop，其中k是一个整数。 （输入保证同时在队列或栈中的数不会超过100个）

**输出**

对每组测试数据输出两行，正常情况下，第一行是队列中从左到右存的数字，第二行是栈中从左到右存的数字。若操作过程中队列或栈已空仍然收到pop，则输出error。输出应该共2*m行。

**样例输入**

```
2
4
push 1
push 3
pop
push 5
1
pop
```

**样例输出**

```
3 5
1 5
error
error
```

```python
m=int(input())
for _ in range(m):
    error=False
    n=int(input())
    queue=[]
    stack=[]
    for i in range(n):
        a=input()
        if a[-1].isdigit():
            b=int(a.split()[-1])
            queue.append(b)
            stack.append(b)
        else:
            if not queue:
                error=True
            else:
                queue.pop(0)
                stack.pop(-1)
    if error:
        print("error")
        print("error")
    else:
        print(*queue)
        print(*stack)
```



#### 波兰表达式（栈）

```python
line=input().split()
stack=[]
for char in line[::-1]:
    if char not in "+ - * /":
        stack.append(float(char))
    else:
        a=stack.pop(-1)
        b=stack.pop(-1)
        if char=="+":
            stack.append(a+b)
        elif char=="-":
            stack.append(a-b)
        elif char=="*":
            stack.append(a*b)
        else:
            stack.append(a/b)
print("{:.6f}".format(stack[0]))

#主要思路：从后面开始读，碰到符号则前二已读数字进行运算
#float(),print("{:.6f}".format(float型数据))
```



#### 22068: 合法出栈序列

**描述**

给定一个由大小写字母和数字构成的，没有重复字符的长度不超过62的字符串x，现在要将该字符串的字符依次压入栈中，然后再全部弹出。

要求左边的字符一定比右边的字符先入栈，出栈顺序无要求。

再给定若干字符串，对每个字符串，判断其是否是可能的x中的字符的出栈序列。

**输入**

第一行是原始字符串x
后面有若干行(不超过50行)，每行一个字符串，所有字符串长度不超过100

**输出**

对除第一行以外的每个字符串，判断其是否是可能的出栈序列。如果是，输出"YES"，否则，输出"NO"

**样例输入**

```
abc
abc
bca
cab
```

**样例输出**

```
YES
YES
NO
```

```python
def check():
    if len(inp) != len(y):
        return False
    x=list(inp)
    for char in y:
        while x and (len(stack)==0 or stack[-1]!=char):
            stack.append(x.pop(0))
        if not x and stack[-1]!=char:
            return False
        else:
            stack.pop(-1)
    return True
inp=input()
while True:
    try:
        y=list(input())
        stack=[]
        if check():
            print("YES")
        else:
            print("NO")
    except EOFError:
        break
#主要思路：找完入栈原因、循环方式、return条件
```



#### 03704:扩号匹配问题

**描述**

在某个字符串（长度不超过100）中有左括号、右括号和大小写字母；规定（与常见的算数式子一样）任何一个左括号都从内到外与在它右边且距离最近的右括号匹配。写一个程序，找到无法匹配的左括号和右括号，输出原来字符串，并在下一行标出不能匹配的括号。不能匹配的左括号用"$"标注,不能匹配的右括号用"?"标注.

**输入**

输入包括多组数据，每组数据一行，包含一个字符串，只包含左右括号和大小写字母

**输出**

对每组输出数据，输出两行，第一行包含原始输入字符，第二行由"$","?"和空格组成，"$"和"?"表示与之对应的左括号和右括号不能匹配。

**样例输入**

```
((ABCD(x)
)(rttyy())sss)(
```

**样例输出**

```
((ABCD(x)
$$
)(rttyy())sss)(
?            ?$
```

```python
def check(n):
    ans=[" "]*len(n)
    for i in range(len(n)):
        if n[i]=="(":
            stack.append(n[i])
            ans[i]="("
        elif n[i]==")":
            if stack:
                stack.pop()
                b=ans[::-1]
                b[b.index("(")]=" "
                ans=b[::-1]
            else:
                ans[i]="?"
    for j in range(len(ans)):
        if ans[j]=="(":
            ans[j]="$"
    return ans
while True:
    try:
        l=input()
        n=list(l)
        stack=[]
        answer=check(n)
        print(l)
        print("".join(answer))
    except EOFError:
        break
```



#### 进制转换

```python
#二进制转换
def binary(x):
    stack=[]
    while x>0:
        stack.append(str(x%2))
        x=x//2
    return "".join(stack[::-1])

print(int(binary(int(input()))))

#十进制到八进制
def oct_(x):
    stack=[]
    while x>0:
        stack.append(str(x%8))
        x=x//8
    return "".join(stack[::-1])

print(int(oct_(int(input()))))
```



#### 中序转后序（栈）

```python
def postfix(x):
    dic={"(":1,")":1,"+":2,"-":2,"*":3,"/":3}
    stack=[]
    ans=[]
    number=""
    for char in x:
        if char.isnumeric() or char==".":
            number+=char
        else:
            if number:
                num=float(number)
                ans.append(int(num) if num.is_integer() else num)
                number=""
            if char=="(":
                stack.append(char)
            elif char==")":
                while stack and dic[stack[-1]]>1:
                    ans.append(stack.pop())
                stack.pop()
            elif char in "+ - * /":
                while stack and dic[stack[-1]]>=dic[char]:
                    ans.append(stack.pop())
                stack.append(char)       
    if number:
        num=float(number)
        ans.append(int(num) if num.is_integer() else num)        
    while stack:
        ans.append(stack.pop())  
    return ans
n=int(input())
for _ in range(n):
    a=input()
    print(" ".join(str(c) for c in postfix(a)))
    
```



#### 02754: 八皇后(栈，回溯，dfs)

```python
def queen_stack(n):
    stack = []  # 用于保存状态的栈
    solutions = [] # 存储所有解决方案的列表
    stack.append((0, []))  # 起点（0行，0可能）
    while stack:
        row, cols = stack.pop() 
        # 从栈中取出最后获得的可行性，再去看加了下一行的可能性并添加
        if row == n:    # 找到一个合法解决方案
            solutions.append(cols)
        else:
            for col in range(n):  #固定行，看每列是否可行
                if is_valid(row, col, cols): # 检查当前位置是否合法
                    stack.append((row + 1, cols + [col]))
    return solutions
#solutions里的添加是逆序的，as像第一行8个可能，then拿最后皇后在（0，7）找第二个皇后可能then拿第二皇后最后可能找第三皇后。。。
def is_valid(row, col, queens):
    for r in range(row):
        if queens[r] == col or abs(row - r) == abs(col - queens[r]):
            return False
    return True
#行已固定一行一个皇后，so看queens[r] == col判断是否撞列， 看abs(row - r) == abs(col - queens[r])看是否对角线

# 获取第 b 个皇后串
def get_queen_string(b):
    solutions = queen_stack(8)
    if b > len(solutions):
        return None
    b = len(solutions) + 1 - b   #因为solution逆序
    queen_string = ''.join(str(col + 1) for col in solutions[b - 1])
    return queen_string
test_cases = int(input())  # 输入的测试数据组数
for _ in range(test_cases):
    b = int(input())  # 输入的 b 值
    queen_string = get_queen_string(b)
    print(queen_string)
```



#### 02746: 约瑟夫问题(queue)

**描述**

约瑟夫问题：有ｎ只猴子，按顺时针方向围成一圈选大王（编号从１到ｎ），从第１号开始报数，一直数到ｍ，数到ｍ的猴子退出圈外，剩下的猴子再接着从1开始报数。就这样，直到圈内只剩下一只猴子时，这个猴子就是猴王，编程求输入ｎ，ｍ后，输出最后猴王的编号。

**输入**

每行是用空格分开的两个整数，第一个是 n, 第二个是 m ( 0 < m,n <=300)。

最后一行是：0 0

**输出**

对于每行输入数据（最后一行除外)，输出数据也是一行，即最后猴王的编号

**样例输入**

```
6 2
12 4
8 3
0 0
```

**样例输出**

```
5
1
7
```

```python
#my method
while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    mon=[x for x in range(1,n+1)]
    nex=0
    while len(mon)>1:
        ind=m+nex-1
        nex=ind%(len(mon))
        mon.pop(ind%(len(mon)))
    print(mon[0])

#用list的pop和append形成循环
while True:
    n,m=map(int,input().split())
    if {n,m}=={0}:
        break
    mon=[x for x in range(1,n+1)]
    nex=0
    while len(mon)>1:
        for i in range(m-1):
            mon.append(mon.pop(0))
        mon.pop(0)
    print(mon[0])
```



#### 05902: 双端队列

**描述**

定义一个双端队列，进队操作与普通队列一样，从队尾进入。出队操作既可以从队头，也可以从队尾。编程实现这个数据结构。

**输入** 

第一行输入一个整数t，代表测试数据的组数。

 每组数据的第一行输入一个整数n，表示操作的次数。 

接着输入n行，每行对应一个操作，首先输入一个整数type。 

当type=1，进队操作，接着输入一个整数x，表示进入队列的元素。 

当type=2，出队操作，接着输入一个整数c，c=0代表从队头出队，c=1代表从队尾出队。

 n <= 1000

**输出** 

对于每组测试数据，输出执行完所有的操作后队列中剩余的元素,元素之间用空格隔开，按队头到队尾的顺序输出，占一行。如果队列中已经没有任何的元素，输出NULL。

**样例输入**

```
2
5
1 2
1 3
1 4
2 0
2 1
6
1 1
1 2
1 3
2 0
2 1
2 0
```

**样例输出**

```
3
NULL
```

```python
#1. list实现deque
def op(deque):
    if x==1:
        deque.append(y)
    else:
        if y==0:
            deque.pop(0)
        else:
            deque.pop()
    return deque
t=int(input())
for _ in range(t):
    n=int(input())
    deque=[]
    for _ in range(n):
        x,y=map(int,input().split())
        ans=op(deque)
    if ans:
        print(*ans)
    else:
        print("NULL")
    
#2. import deque
from collections import deque
for _ in range(int(input())):
    n=int(input())
    q=deque([])
    for i in range(n):
        a,b=map(int,input().split())
        if a==1:
            q.append(b)
        else:
            if b==0:
                q.popleft()
            else:
                q.pop()
    if q:
        print(*q)
    else:
        print('NULL')
```



#### 回文数字（deque)

```python
#可以不用把整个倒反就能看出是不是回文，即可以省时间和空间
from collections import deque
def check(a):
    n=str(a)
    num=deque(n)
    while len(num)>1:        #while num 不对！
        if num.popleft()!=num.pop():
            return "NO"
    return "YES"
while True:
    try:
        a=int(input())
        print(check(a))
    except EOFError:
        break
```



#### KMP

KMP（Knuth-Morris-Pratt）算法是一种利用双指针和动态规划的字符串匹配算法。

```python
#lps存储pattern当前位置的最长前缀后缀，如“AABAA":[0,1,0,1,2]
#指针length，i：length为前缀包含的长度（指针位于前缀末端字符），i为遍历pattern的指针
#compute_lps找出pattern各个字符的前缀长并储存
def compute_lps(pattern):
    m=len(pattern)
    lps=[0]*m
    length=0
    for i in range(1,m):
        while length>0 and pattern[i]!=pattern[length]:
            length=lps[length-1]
        if pattern[i]==pattern[length]: 
            #pattern[i]的字符与上个字符的前缀的后一个字符相等，该段前缀长+1
            length+=1
        lps[i]=length
    return lps
 #while loop:"AABAAC"中lps=[0,1,0,1,2,0]:当i指向C,length此时因上个字符而为2（即lps[i-1])
 #故pattern[5]!=pattern[2] (C!=B) ~length往前一步，看pattern[i]是否有它的前缀
 #如上C！=A,length再-1=0 跳出循环
 
def kmp_search(text,pattern):
    n=len(text)
    m=len(pattern)
    if m==0:
        return 0
    lps=compute_lps(pattern)
    matches=[]
    j=0                            #pattern索引
    for i in range(n):             #text索引
        while j>0 and text[i]!=pattern[j]:
            j=lps[j-1]                   
        if text[i]==pattern[j]:
            j+=1
        if j==m:
            matches.append(i-j+1)
            j=lps[j-1]         #match的最后一个字符开始再看pat出现到哪里（pat里面也会重复，所以前缀不一定是从pat第一个字符开始看
    return matches

#pat:"ABBAAB"~lps:[0,0,0,1,1,2],text:"ABBAABBAABBAAB"~match:[0,4,8]
    
text=input()
pattern=input()
index=kmp_search(text,pattern)
print("pos matched:",index)
#"ABABABABCABABABABCABABABABC","ABABCABAB"
#pos matched： [4, 13]
```



#### 06640: 倒排索引

**描述**

给定一些文档，要求求出某些单词的倒排表。

对于一个单词，它的倒排表的内容为出现这个单词的文档编号。

**输入**

第一行包含一个数N，1 <= N <= 1000，表示文档数。 接下来N行，每行第一个数ci，表示第i个文档的单词数。接下来跟着ci个用空格隔开的单词，表示第i个文档包含的单词。文档从1开始编号。1 <= ci <= 100。 接下来一行包含一个数M，1 <= M <= 1000，表示查询数。 接下来M行，每行包含一个单词，表示需要输出倒排表的单词。 每个单词全部由小写字母组成，长度不会超过256个字符，大多数不会超过10个字符。

**输出**

对于每一个进行查询的单词，输出它的倒排表，文档编号按从小到大排序。 如果倒排表为空，输出"NOT FOUND"。

**样例输入**

```
3
2 hello world
4 the world is great
2 great news
4
hello
world
great
pku
```

**样例输出**

```
1
1 2
2 3
NOT FOUND
```

```python
#method my
n=int(input())
dic={}
for ind in range(1,n+1):   #ind=doc_index
    w=input().split( )
    for word in w[1:]:
        if word not in dic:
            dic[word]=set()  #不能用list as同一行可能有重复word
        dic[word].add(ind)
m=int(input())
answer=[]
for _ in range(m):
    check=input()
    if check in dic:  #直接判断有没有，不要分两step
        answer.append(" ".join(map(str,sorted(dic[check]))))   #sorted!!
    else:
        answer.append("NOT FOUND")
for a in answer:
    print(a)

#建议代码
def main():
    import sys       #sys用于处理大量输入
    input = sys.stdin.read #一次性读入所有输入
    data = input().splitlines() #把输入逐行放进list
    n = int(data[0])
    index = 1
    inverted_index = {}   # 构建倒排索引
    for i in range(1, n + 1):
        parts = data[index].split()
        doc_id = i
        num_words = int(parts[0])
        words = parts[1:num_words + 1]
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            inverted_index[word].append(doc_id)
        index += 1
    m = int(data[index])
    index += 1
    results = []
    # 查询倒排索引
    for _ in range(m):
        query = data[index]
        index += 1
        if query in inverted_index:
            results.append(" ".join(map(str, sorted(inverted_index[query]))))
        else:
            results.append("NOT FOUND")
    # 输出查询结果
    for result in results:
        print(result)
if __name__ == "__main__":
    main()
```



#### 04093: 倒排索引查询

**描述**

现在已经对一些文档求出了倒排索引，对于一些词得出了这些词在哪些文档中出现的列表。

要求对于倒排索引实现一些简单的查询，即查询某些词同时出现，或者有些词出现有些词不出现的文档有哪些。

**输入**

第一行包含一个数N，1 <= N <= 100，表示倒排索引表的数目。 接下来N行，每行第一个数ci，表示这个词出现在了多少个文档中。接下来跟着ci个数，表示出现在的文档编号，编号不一定有序。1 <= ci <= 1000，文档编号为32位整数。 接下来一行包含一个数M，1 <= M <= 100，表示查询的数目。 接下来M行每行N个数，每个数表示这个词要不要出现，1表示出现，-1表示不出现，0表示无所谓。数据保证每行至少出现一个1。

**输出**

共M行，每行对应一个查询。输出查询到的文档编号，按照编号升序输出。 如果查不到任何文档，输出"NOT FOUND"。

**样例输入**

```
3
3 1 2 3
1 2
1 3
3
1 1 1
1 -1 0
1 -1 -1
```

**样例输出**

```
NOT FOUND
1 3
1
```

```python
import sys
input = sys.stdin.read
data = input().split()
index = 0
N = int(data[index])
index += 1
word_doc=[]
for _ in range(N):
    ci=int(data[index])
    index+=1
    doc=sorted(map(int,data[index:index+ci])) #词所在的表们
    index+=ci
    word_doc.append(doc)
m=int(data[index])
index+=1
result=[]
for _ in range(m):
    query=list(map(int,data[index:index+N]))
    index+=N
    inc_doc=[]         #需要每个呈现词都有出现该表
    exc_doc=set()      #只要表出现在exc那答案直接不会有该表
    for i in range(N):
        if query[i]==1:
            inc_doc.append(word_doc[i])
        elif query[i]==-1:
            exc_doc.update(word_doc[i])
    if inc_doc:
        result_set=set(inc_doc[0])
        for d in inc_doc[1:]:
            result_set.intersection_update(d)   
            #找inc里每个词的并集（共同表）
        result_set.difference_update(exc_doc)   
        #把exc的结果加进去，即减掉inc和exc共同的表
        final_doc=sorted(result_set)
        result.append(" ".join(map(str,final_doc)) if final_doc else "NOT FOUND")   #把每次查询结果set转换为str并储存
    else:
        result.append("NOT FOUND")
for r in result:
    print(r)
```



#### 02766: 最大子矩阵(dp)

**描述**

已知矩阵的大小定义为矩阵中所有元素的和。给定一个矩阵，你的任务是找到最大的非空(大小至少是1 * 1)子矩阵。

比如，如下4 * 4的矩阵

0 -2 -7 0 

9 2 -6 2 

-4 1 -4 1 

-1 8 0 -2

的最大子矩阵是

9 2 

-4 1 

-1 8

这个子矩阵的大小是15。

**输入**

输入是一个N * N的矩阵。输入的第一行给出N (0 < N <= 100)。再后面的若干行中，依次（首先从左到右给出第一行的N个整数，再从左到右给出第二行的N个整数……）给出矩阵中的N2个整数，整数之间由空白字符分隔（空格或者空行）。已知矩阵中整数的范围都在[-127, 127]。

**输出**

输出最大子矩阵的大小。

**样例输入**

```
4
0 -2 -7 0 9 2 -6 2
-4 1 -4  1 -1

8  0 -2
```

**样例输出**

```
15
```

```python
'''
为了找到最大的非空子矩阵，可以使用动态规划中的Kadane算法进行扩展来处理二维矩阵。
基本思路是将二维问题转化为一维问题：可以计算出从第i行到第j行的列的累计和，
这样就得到了一个一维数组。然后对这个一维数组应用Kadane算法，找到最大的子数组和。
通过遍历所有可能的行组合，我们可以找到最大的子矩阵。
'''
def max_submatrix(matrix):
    def kadane(arr):
      	# max_ending_here 用于追踪到当前元素为止包含当前元素的最大子数组和。
        # max_so_far 用于存储迄今为止遇到的最大子数组和。
        max_end_here = max_so_far = arr[0]
        for x in arr[1:]:
          	# 对于每个新元素，我们决定是开始一个新的子数组（仅包含当前元素 x），
            # 还是将当前元素添加到现有的子数组中。这一步是 Kadane 算法的核心。
            max_end_here = max(x, max_end_here + x)
            max_so_far = max(max_so_far, max_end_here)
        return max_so_far
    rows = len(matrix)
    cols = len(matrix[0])
    max_sum = float('-inf')
    for left in range(cols):
        temp = [0] * rows
        for right in range(left, cols):
            for row in range(rows):
                temp[row] += matrix[row][right]
            max_sum = max(max_sum, kadane(temp))
    return max_sum
n = int(input())
nums = []
while len(nums) < n * n:
    nums.extend(input().split())
matrix = [list(map(int, nums[i * n:(i + 1) * n])) for i in range(n)]
max_sum = max_submatrix(matrix)
print(max_sum)
```



#### 工具

int(str,n)  将字符串`str`转换为`n`进制的整数。

for key,value in dict.items()   遍历字典的键值对。

for index,value in enumerate(list)  枚举列表，提供元素及其索引。

dict.get(key,default)   从字典中获取键对应的值，如果键不存在，则返回默认值`default`。

list(zip(a,b))  将两个列表元素一一配对，生成元组的列表。

math.pow(m,n)   计算`m`的`n`次幂。

math.log(m,n)   计算以`n`为底的`m`的对数。

lrucache    

```py
from functools import lru_cache
@lru_cache(maxsize=None)
```

bisect

```python
import bisect
# 创建一个有序列表
sorted_list = [1, 3, 4, 4, 5, 7]
# 使用bisect_left查找插入点
position = bisect.bisect_left(sorted_list, 4)
print(position)  # 输出: 2
# 使用bisect_right查找插入点
position = bisect.bisect_right(sorted_list, 4)
print(position)  # 输出: 4
# 使用insort_left插入元素
bisect.insort_left(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 5, 7]
# 使用insort_right插入元素
bisect.insort_right(sorted_list, 4)
print(sorted_list)  # 输出: [1, 3, 4, 4, 4, 4, 5, 7]
```

字符串

1. `str.lstrip() / str.rstrip()`: 移除字符串左侧/右侧的空白字符。

2. `str.find(sub)`: 返回子字符串`sub`在字符串中首次出现的索引，如果未找到，则返回-1。

3. `str.replace(old, new)`: 将字符串中的`old`子字符串替换为`new`。

4. `str.startswith(prefix) / str.endswith(suffix)`: 检查字符串是否以`prefix`开头或以`suffix`结尾。

5. `str.isalpha() / str.isdigit() / str.isalnum()`: 检查字符串是否全部由字母/数字/字母和数字组成。

   6.`str.title()`：每个单词首字母大写。



counter：计数

```python
from collections import Counter
# 创建一个Counter对象
count = Counter(['apple', 'banana', 'apple', 'orange', 'banana', 'apple'])
# 输出Counter对象
print(count)  # 输出: Counter({'apple': 3, 'banana': 2, 'orange': 1})
# 访问单个元素的计数
print(count['apple'])  # 输出: 3
# 访问不存在的元素返回0
print(count['grape'])  # 输出: 0
# 添加元素
count.update(['grape', 'apple'])
print(count)  # 输出: Counter({'apple': 4, 'banana': 2, 'orange': 1, 'grape': 1})
```



permutations：全排列

```python
from itertools import permutations
# 创建一个可迭代对象的排列
perm = permutations([1, 2, 3])
# 打印所有排列
for p in perm:
    print(p)
# 输出: (1, 2, 3)，(1, 3, 2)，(2, 1, 3)，(2, 3, 1)，(3, 1, 2)，(3, 2, 1)
```



combinations：组合

```python
from itertools import combinations
# 创建一个可迭代对象的组合
comb = combinations([1, 2, 3], 2)
# 打印所有组合
for c in comb:
    print(c)
# 输出: (1, 2)，(1, 3)，(2, 3)
```



reduce：累次运算

```python
from functools import reduce
# 使用reduce计算列表元素的乘积
product = reduce(lambda x, y: x * y, [1, 2, 3, 4])
print(product)  # 输出: 24
```



product：笛卡尔积

```python
from itertools import product
# 创建两个可迭代对象的笛卡尔积
prod = product([1, 2], ['a', 'b'])
# 打印所有笛卡尔积对
for p in prod:
    print(p)
# 输出: (1, 'a')，(1, 'b')，(2, 'a')，(2, 'b')
```



一些容易忘记的操作

```python
a = [1, 2, 3, 4, 5]
print(*a) #out:1 2 3 4 5
print(*a, sep = ", ") #out:1,2,3,4,5
print(*a, sep = "\n") #out:1 （下一行）2…
print str(a)[1:-1] #out:1,2,3,4,5
print(“”.join(a)) #for 元素all str
print(l[:]) 或 print(l[0:]) 或print(l[0:len(l)])
print(i, end=“ “) for i in a

print(“str”,int) # ==str int
print(“str”+str(int)) # ==strint
print(f“str ={int}+{int}”) # 在需要插入int的地方放{int}
print(“%d+%d=%d” %(int,int,int)) #占位符%d
print(“{}+{}={}”.format(int,int,int))
print(“{:.2f}”.format(int))
print(“%.2f”%int 或者 float)

list.sort(reverse=True) #list逆序排
list.reverse #list颠倒
str[::-1] #str颠倒
int(str,base) #把str换成以base为底的值，base默认10

x=a.split(“.”)[0] y=a.split(“.”)[1] #分割str格式的小数(a):x=整数，y=小数后

for x in enumerate(b) : #产生tuple(index,value)
	ix,vx=x #unp
```

