# Assignment #6: "树"算：Huffman,BinHeap,BST,AVL,DisjointSet

Updated 2214 GMT+8 March 24, 2024

2024 spring, Complied by 石芯洁 数学科学学院==同学的姓名、院系==



**说明：**

1）这次作业内容不简单，耗时长的话直接参考题解。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c)

Python编程环境：Spyder（Python 3.11)

## 1. 题目

### 22275: 二叉搜索树的遍历

http://cs101.openjudge.cn/practice/22275/



思路：



代码

```python
# 
class Node:
    def __init__(self,val,left=None,right=None):
        self.val=val
        self.left=left
        self.right=right

def build(preorder):
    if len(preorder)==0:
        return None
    node=Node(preorder[0])
    idx=len(preorder)
    for i in range(1,len(preorder)):
        if preorder[i]>preorder[0]:
            idx=i
            break
    node.left=build(preorder[1:idx])
    node.right=build(preorder[idx:])
    
    return node

def postorder(node):
    if node is None:
        return []
    
    output=[]
    output.extend(postorder(node.left))
    output.extend(postorder(node.right))
    output.append(str(node.val))
    return output

n=int(input())
preorder=list(map(int,input().split()))
print(" ".join(postorder(build(preorder))))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 093953.png)



### 05455: 二叉搜索树的层次遍历

http://cs101.openjudge.cn/practice/05455/



思路：



代码

```python
# 
class Node:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def build(node,value):
    if node==None:
        return Node(value)
    if node.value > value:
        node.left=build(node.left,value)
    elif node.value < value:
        node.right=build(node.right,value)
    return node

def ordertravel(root):
    queue=[root]
    ans=[]
    while queue:
        node=queue.pop(0)
        ans.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return ans

line=list(map(int,input().strip().split()))
line=list(dict.fromkeys(line))
root=None
for num in line:
    root=build(root,num)
ans=ordertravel(root)
print(" ".join(map(str,ans)))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 135745.png)



### 04078: 实现堆结构

http://cs101.openjudge.cn/practice/04078/

练习自己写个BinHeap。当然机考时候，如果遇到这样题目，直接import heapq。手搓栈、队列、堆、AVL等，考试前需要搓个遍。



思路：



代码

```python
# 
n=int(input())
lis=[]
for _ in range(n):
    inp=input().strip()
    if inp[0]=="1":
        typ,u=map(int,inp.split())
        lis.append(u)
    else:
        typ=2
        if lis:
            output=min(lis)
            lis.remove(output)
            print(output)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 153803.png)



### 22161: 哈夫曼编码树

http://cs101.openjudge.cn/practice/22161/



思路：



代码

```python
# 
import heapq

class Node:
    def __init__(self, weight, char=None):
        self.weight = weight
        self.char = char
        self.left = None
        self.right = None

    def __lt__(self, other):
        if self.weight == other.weight:
            return self.char < other.char
        return self.weight < other.weight

def build_huffman_tree(characters):
    heap = []
    for char, weight in characters.items():
        heapq.heappush(heap, Node(weight, char))

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = Node(left.weight + right.weight, min(left.char, right.char))
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]

def encode_huffman_tree(root):
    codes = {}

    def traverse(node, code):
        #if node.char:
        if node.left is None and node.right is None:
            codes[node.char] = code
        else:
            traverse(node.left, code + '0')
            traverse(node.right, code + '1')

    traverse(root, '')
    return codes

def huffman_encoding(codes, string):
    encoded = ''
    for char in string:
        encoded += codes[char]
    return encoded

def huffman_decoding(root, encoded_string):
    decoded = ''
    node = root
    for bit in encoded_string:
        if bit == '0':
            node = node.left
        else:
            node = node.right

        if node.left is None and node.right is None:
            decoded += node.char
            node = root
    return decoded

n = int(input())
characters = {}
for _ in range(n):
    char, weight = input().split()
    characters[char] = int(weight)

huffman_tree = build_huffman_tree(characters)

codes = encode_huffman_tree(huffman_tree)

strings = []
while True:
    try:
        line = input()
        strings.append(line)

    except EOFError:
        break

results = []
for string in strings:
    if string[0] in ('0','1'):
        results.append(huffman_decoding(huffman_tree, string))
    else:
        results.append(huffman_encoding(codes, string))

for result in results:
    print(result)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 161015.png)



### 晴问9.5: 平衡二叉树的建立

https://sunnywhy.com/sfbj/9/5/359



思路：



代码

```python
# 
class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.height = 1

class AVL:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self.root = self._insert(value, self.root)

    def _insert(self, value, node):
        if not node:
            return Node(value)
        elif value < node.value:
            node.left = self._insert(value, node.left)
        else:
            node.right = self._insert(value, node.right)

        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))

        balance = self._get_balance(node)

        if balance > 1:
            if value < node.left.value:	        #LL
                return self._rotate_right(node)
            else:	                            #LR
                node.left = self._rotate_left(node.left)
                return self._rotate_right(node)

        if balance < -1:
            if value > node.right.value:	    #RR
                return self._rotate_left(node)
            else:	                            #RL
                node.right = self._rotate_right(node.right)
                return self._rotate_left(node)

        return node

    def _get_height(self, node):
        if not node:
            return 0
        return node.height

    def _get_balance(self, node):
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _rotate_left(self, z):
        y = z.right
        T2 = y.left
        y.left = z
        z.right = T2
        z.height = 1 + max(self._get_height(z.left), self._get_height(z.right))
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        return y

    def _rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        y.height = 1 + max(self._get_height(y.left), self._get_height(y.right))
        x.height = 1 + max(self._get_height(x.left), self._get_height(x.right))
        return x

    def preorder(self):
        return self._preorder(self.root)

    def _preorder(self, node):
        if not node:
            return []
        return [node.value] + self._preorder(node.left) + self._preorder(node.right)

n = int(input().strip())
sequence = list(map(int, input().strip().split()))

avl = AVL()
for value in sequence:
    avl.insert(value)

print(' '.join(map(str, avl.preorder())))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 175429.png)



### 02524: 宗教信仰

http://cs101.openjudge.cn/practice/02524/



思路：



代码

```python
# 
class DisjointSetUnion:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

def estimate_religions(n, m, edges):
    dsu = DisjointSetUnion(n)
    for edge in edges:
        dsu.union(edge[0] - 1, edge[1] - 1)
    
    religions = set()
    for i in range(n):
        religions.add(dsu.find(i))
    
    return len(religions)

case_number = 0
while True:
    n, m = map(int, input().split())
    if n == 0 and m == 0:
        break
    case_number += 1
    edges = [tuple(map(int, input().split())) for _ in range(m)]
    religions_limit = estimate_religions(n, m, edges)
    print(f"Case {case_number}: {religions_limit}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-02 231559.png)



## 2. 学习总结和收获

这次题目好难，基本是依靠题解和ChatGPT的解释来理解及尝试。





==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





