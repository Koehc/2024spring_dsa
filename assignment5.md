# Assignment #5: "树"算：概念、表示、解析、遍历

Updated 2124 GMT+8 March 17, 2024

2024 spring, Complied by  石芯洁 数学科学学院==同学的姓名、院系==



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

### 27638: 求二叉树的高度和叶子数目

http://cs101.openjudge.cn/practice/27638/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def height(root):
    if root is None:
        return -1
    left = height(root.left)
    right = height(root.right)
    
    return max(left, right) + 1

def leaf_counts(root):
    if root is None:
        return 0
    if root.left is None and root.right is None:
        return 1
    return leaf_counts(root.left) + leaf_counts(root.right) 

if __name__ == "__main__":
    n = int(input())
    nodes = [TreeNode() for _ in range(n)]
    has_parent = [False] * n  

    for i in range(n):
        left_index, right_index = map(int, input().split())
        if left_index != -1:
            nodes[i].left = nodes[left_index]
            has_parent[left_index] = True
        if right_index != -1:
            nodes[i].right = nodes[right_index]
            has_parent[right_index] = True
            
        root_index = has_parent.index(False)
        root = nodes[root_index]

    height = height(root)
    leaves = leaf_counts(root)

    print(height,leaves)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 140846.png)



### 24729: 括号嵌套树

http://cs101.openjudge.cn/practice/24729/



思路：



代码

```python
# 
class TreeNode():
    def __init__(self,value):
        self.value = value
        self.children = []

def parse_tree(s):
    stack = []
    node = None
    for char in s:
        if char.isalpha():
            node = TreeNode(char)
            if stack:
                stack[-1].children.append(node)
        elif char == '(': 
            if node: 
                stack.append(node)
                node = None 
        elif char == ')':
            if stack: 
                node = stack.pop()
    return node

def preorder(node):
    output = [node.value] 
    for child in node.children:
        output.extend(preorder(child))
    return ''.join(output)

def postorder(node):
    output =[]
    for child in node.children:
        output.extend(postorder(child))
    output.append(node.value)
    return ''.join(output)

s = input().strip() 
s = ''.join(s.split())
root = parse_tree(s)
if root:
    print(preorder(root))
    print(postorder(root))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 151448.png)



### 02775: 文件结构“图”

http://cs101.openjudge.cn/practice/02775/



思路：



代码

```python
# 
class Dir:
    def __init__(self, name):
        self.name = name
        self.dirs = []
        self.files = []

    def get_graph(self):
        g = [self.name]
        for d in self.dirs:
            sub = d.get_graph()
            g.extend(["|     " + s for s in sub])
        for f in sorted(self.files):
            g.append(f)
        return g

n = 0
try:
    while True:
        n += 1
        stack = [Dir('ROOT')]
        while True:
            s = input()
            if s == '*':
                break
            elif s == "#":
                raise EOFError

            if s[0] == "f":
                stack[-1].files.append(s)
            elif s[0] == 'd':
                stack.append(Dir(s))
                stack[-2].dirs.append(stack[-1])
            else:
                stack.pop()

        print(f"DATA SET {n}:")
        print(*stack[0].get_graph(), sep='\n')
        print()
except EOFError:
    pass
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 162235.png)



### 25140: 根据后序表达式建立队列表达式

http://cs101.openjudge.cn/practice/25140/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(postfix):
    stack = []
    for char in postfix:
        node = TreeNode(char)
        if char.isupper():
            node.right = stack.pop()
            node.left = stack.pop()
        stack.append(node)
    return stack[0]

def level_order_traversal(root):
    queue = [root]
    traversal = []
    while queue:
        node = queue.pop(0)
        traversal.append(node.value)
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return traversal

n = int(input().strip())
for _ in range(n):
    postfix = input().strip()
    root = build_tree(postfix)
    queue_expression = level_order_traversal(root)[::-1]
    print(''.join(queue_expression))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 164442.png)



### 24750: 根据二叉树中后序序列建树

http://cs101.openjudge.cn/practice/24750/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def buildTree(inorder, postorder):
    if not inorder or not postorder:
        return None
    root_val = postorder.pop()
    root = TreeNode(root_val)
    root_index = inorder.index(root_val)
    root.right = buildTree(inorder[root_index + 1:], postorder)
    root.left = buildTree(inorder[:root_index], postorder)
    return root

def preorderTraversal(root):
    result = []
    if root:
        result.append(root.val)
        result.extend(preorderTraversal(root.left))
        result.extend(preorderTraversal(root.right))
    return result

inorder = input().strip()
postorder = input().strip()

root = buildTree(list(inorder), list(postorder))
print(''.join(preorderTraversal(root)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 172642.png)



### 22158: 根据二叉树前中序序列建树

http://cs101.openjudge.cn/practice/22158/



思路：



代码

```python
# 
class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

def build_tree(preorder, inorder):
    if not preorder or not inorder:
        return None
    root_value = preorder[0]
    root = TreeNode(root_value)
    root_index_inorder = inorder.index(root_value)
    root.left = build_tree(preorder[1:1+root_index_inorder], inorder[:root_index_inorder])
    root.right = build_tree(preorder[1+root_index_inorder:], inorder[root_index_inorder+1:])
    return root

def postorder_traversal(root):
    if root is None:
        return ''
    return postorder_traversal(root.left) + postorder_traversal(root.right) + root.value

while True:
    try:
        preorder = input().strip()
        inorder = input().strip()
        root = build_tree(preorder, inorder)
        print(postorder_traversal(root))
    except EOFError:
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-26 174723.png)



## 2. 学习总结和收获

之前几乎对二叉树完全不了解，通过这次功课才稍微了解了二叉树。虽然大部分还是依靠ChatGPT完成。

==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





