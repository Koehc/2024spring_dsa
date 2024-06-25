# Assignment #9: 图论：遍历，及 树算

Updated 1739 GMT+8 Apr 14, 2024

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

### 04081: 树的转换

http://cs101.openjudge.cn/dsapre/04081/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self):
        self.children=[]
        self.first_child=None
        self.next_sib=None
        
def build(data):
    root=TreeNode()
    stack=[root]
    depth=0
    for act in data:
        cur_node=stack[-1]
        if act=="d":
            new_node=TreeNode()
            if not cur_node.children:
                cur_node.first_child=new_node
            else:
                cur_node.children[-1].next_sib=new_node
            cur_node.children.append(new_node)
            stack.append(new_node)
            depth=max(depth,len(stack)-1)
        else:
            stack.pop()
    return root,depth

def cal_h(node):
    if not node:
        return -1
    return max(cal_h(node.first_child),cal_h(node.next_sib))+1

data=input()
root,h_ori=build(data)
h_bin=cal_h(root)
print(f"{h_ori} => {h_bin}")
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 144652.png)



### 08581: 扩展二叉树

http://cs101.openjudge.cn/dsapre/08581/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self,value):
        self.value=value
        self.left=None
        self.right=None

def build(data):
    if not data:
        return None
    value=data.pop()
    if value==".":
        return None
    root=TreeNode(value)
    root.left=build(data)
    root.right=build(data)
    return root

def inorder(root):
    if not root:
        return[]
    left=inorder(root.left)
    right=inorder(root.right)
    return left+[root.value]+right

def postorder(root):
    if not root:
        return []
    left=postorder(root.left)
    right=postorder(root.right)
    return left+right+[root.value]

data=list(input())
root=build(data[::-1])
inorder_result=inorder(root)
postorder_result=postorder(root)
print("".join(inorder_result))
print("".join(postorder_result))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 153519.png)



### 22067: 快速堆猪

http://cs101.openjudge.cn/practice/22067/



思路：



代码

```python
# 
stack=[]
min_each=[]            #i只猪在stack时的最轻体重=min_each[i]

while True:
    try:
        s=input().split()
        if s[0]=="pop":
            if stack:
                stack.pop()
                min_each.pop()
        elif s[0]=="min":
            if stack:
                print(min_each[-1])
        else:
            mass=int(s[1])
            stack.append(mass)
            if not min_each:
                min_each.append(mass)
            else:
                least=min_each[-1]
                min_each.append(min(least,mass))
        
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 155623.png)



### 04123: 马走日

dfs, http://cs101.openjudge.cn/practice/04123



思路：



代码

```python
# 
def is_valid_move(board_size, visited, row, col):
    return 0 <= row < board_size[0] and 0 <= col < board_size[1] and not visited[row][col]

def knight_tour(board_size, start_row, start_col):
    moves = [(2, 1), (2, -1), (-2, 1), (-2, -1),
             (1, 2), (1, -2), (-1, 2), (-1, -2)]

    visited = [[False] * board_size[1] for _ in range(board_size[0])]
    visited[start_row][start_col] = True
    count = [0]

    def dfs(row, col, visited, count):
        if all(all(row) for row in visited):
            count[0] += 1
            return

        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            if is_valid_move(board_size, visited, next_row, next_col):
                visited[next_row][next_col] = True
                dfs(next_row, next_col, visited, count)
                visited[next_row][next_col] = False

    dfs(start_row, start_col, visited, count)
    return count[0]


T = int(input())

for _ in range(T):
    n, m, x, y = map(int, input().split())
    print(knight_tour((n, m), x, y))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 161146.png)



### 28046: 词梯

bfs, http://cs101.openjudge.cn/practice/28046/



思路：



代码

```python
# 
from collections import defaultdict
dic=defaultdict(list)
n=int(input())
lis=[]
for i in range(n):
    lis.append(input())
    
for word in lis:
    for i in range(len(word)):
        buc=word[:i]+"_"+word[i+1:]
        dic[buc].append(word)

def bfs(start,end,dic):
    queue=[(start,[start])]
    visited=[start]
    while queue:
        curword,curpath=queue.pop(0)
        if curword==end:
            return " ".join(curpath)
        for i in range(len(curword)):
            pos=curword[:i]+"_"+curword[i+1:]
            for nbr in dic[pos]:
                if nbr not in visited:
                    visited.append(nbr)
                    newpath=curpath+[nbr]
                    queue.append((nbr,newpath))
    return "NO"

start,end=map(str,input().split())
print(bfs(start,end,dic))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 163408.png)



### 28050: 骑士周游

dfs, http://cs101.openjudge.cn/practice/28050/



思路：



代码

```python
# 
def is_valid_move(board_size, visited, row, col):
    return 0 <= row < board_size and 0 <= col < board_size and not visited[row][col]

def knight_tour(board_size, start_row, start_col):
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    visited = [[False] * board_size for _ in range(board_size)]
    visited[start_row][start_col] = True

    def get_neighbors(row, col):
        neighbors = []
        for dr, dc in moves:
            next_row, next_col = row + dr, col + dc
            if is_valid_move(board_size, visited, next_row, next_col):
                count = sum(1 for dr, dc in moves if is_valid_move(board_size, visited, next_row + dr, next_col + dc))
                neighbors.append((count, next_row, next_col))
        return neighbors

    def dfs(row, col, count):
        if count == board_size ** 2 - 1:
            return True

        neighbors = get_neighbors(row, col)
        neighbors.sort()

        for _, next_row, next_col in neighbors:
            visited[next_row][next_col] = True
            if dfs(next_row, next_col, count + 1):
                return True
            visited[next_row][next_col] = False

        return False

    return dfs(start_row, start_col, 0)

board_size = int(input())
start_row, start_col = map(int, input().split())
if knight_tour(board_size, start_row, start_col):
    print("success")
else:
    print("fail")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-04-23 171240.png)



## 2. 学习总结和收获

题目有难度，大部分都是依靠题解完成。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





