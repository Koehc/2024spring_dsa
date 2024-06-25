# Assignment #1: 拉齐大家Python水平

Updated 0940 GMT+8 Feb 19, 2024

2024 spring, Complied by 石芯洁 数学科学学院==同学的姓名、院系==



**说明：**

1）数算课程的先修课是计概，由于计概学习中可能使用了不同的编程语言，而数算课程要求Python语言，因此第一周作业练习Python编程。如果有同学坚持使用C/C++，也可以，但是建议也要会Python语言。

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c)

Python编程环境：Spyder（Python 3.11)



## 1. 题目

### 20742: 泰波拿契數

http://cs101.openjudge.cn/practice/20742/



思路：



##### 代码

```python
# 
n=int(input())
m=[-1]*(n+1)
m[0]=0
m[1]=1
m[2]=1

for i in range(n+1):
    if m[i]==-1:
        m[i]=m[i-1]+m[i-2]+m[i-3]

print(m[n])
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102044.png)



### 58A. Chat room

greedy/strings, 1000, http://codeforces.com/problemset/problem/58/A



思路：



##### 代码

```python
# 
s=input()
h="hello"
x=0
for c in s:
    if c==h[x]:
        x+=1
    if x==len(h):
        break
    
print("YES" if x>=len(h) else "NO")
   
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102218.png)



### 118A. String Task

implementation/strings, 1000, http://codeforces.com/problemset/problem/118/A



思路：



##### 代码

```python
# 
def process_string(input_string):
    vowels=[ "a", "o", "y", "e", "u", "i"]
    result=""
    
    for char in input_string:
        char_lower=char.lower()
        if char_lower in vowels:
            continue
        else:
            result+="."+char_lower
    return result
 
input_string=input()
output_string=process_string(input_string)
 
if process_string(input_string):
    print(output_string)
 
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102320.png)



### 22359: Goldbach Conjecture

http://cs101.openjudge.cn/practice/22359/



思路：



##### 代码

```python
# 
n=int(input())

def prime(x):
    for i in range(2,int(x**0.5)+1):
        if x%i==0:
            return False
    return True

for a in range(n//2 +1):
    if prime(a) and prime(n-a):
        print(a ,(n-a))
        break
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102359.png)



### 23563: 多项式时间复杂度

http://cs101.openjudge.cn/practice/23563/



思路：



##### 代码

```python
# 
term=input().split("+")
a=0
for i in term:
    for j in range(len(i)):
        if i[j]=="n":
            if i[j-1] != "0":
                a=max(int(i[j+2:]),a)
                
print(f"n^{a}")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102623.png)



### 24684: 直播计票

http://cs101.openjudge.cn/practice/24684/



思路：



##### 代码

```python
# 
votes=list(map(int,input().split()))

def winner(votes):
    count={}
    mx_count=0
    for vote in votes:
        if vote in count:
            count[vote]+=1
        else:
            count[vote]=1
        mx_count=max(mx_count,count[vote])
    winners=[k for k,v in count.items() if v==mx_count]
    return sorted(winners)

print(*winner(votes))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-02-21 102717.png)



## 2. 学习总结和收获

这次题目较多是上个学期已完成的题目，因此完成较快。同时，也终于有我比较会做的功课了。

==如果作业题目简单，有否额外练习题目，比如：OJ“数算pre每日选做”、CF、LeetCode、洛谷等网站题目。==





