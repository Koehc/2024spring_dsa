# Assignment #3: March月考

Updated 1537 GMT+8 March 6, 2024

2024 spring, Complied by 石芯洁 数学科学学院==同学的姓名、院系==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c)

Python编程环境：Spyder（Python 3.11)



## 1. 题目

**02945: 拦截导弹**

http://cs101.openjudge.cn/practice/02945/



思路：



##### 代码

```python
# 
k=int(input())
data=list(map(int,input().split()))
numbomb=[0]*k              

for i in range(k-1,-1,-1):
    maxbomb=1
    for j in range(k-1,i,-1):                 
        if data[j]<=data[i] and numbomb[j]+1>maxbomb:
            maxbomb=numbomb[j]+1
    numbomb[i]=maxbomb

print(max(numbomb))
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 084450.png)



**04147:汉诺塔问题(Tower of Hanoi)**

http://cs101.openjudge.cn/practice/04147



思路：



##### 代码

```python
# 
def moveone(numdisk:int,init:str,dest:str):
    print("{}:{}->{}".format(numdisk,init,dest))

def move(numdisks,init,temp,dest):
    if numdisks==1:
        moveone(1,init,dest)
    else:
        move(numdisks-1,init,dest,temp)
        moveone(numdisks,init,dest)
        move(numdisks-1,temp,init,dest)

n,a,b,c=input().split()
move(int(n),a,b,c)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 092623.png)



**03253: 约瑟夫问题No.2**

http://cs101.openjudge.cn/practice/03253



思路：



##### 代码

```python
# 
while True:
    n, p, m = map(int, input().split())
    if {n,p,m} == {0}:
        break
    child = [i for i in range(1, n+1)]
    for _ in range(p-1):
        tmp = child.pop(0)
        child.append(tmp)

    index = 0
    ans = []
    while len(child) != 1:
        temp = child.pop(0)
        index += 1
        if index == m:
            index = 0
            ans.append(temp)
            continue
        child.append(temp)

    ans.extend(child)

    print(','.join(map(str, ans)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 162621.png)



**21554:排队做实验 (greedy)v0.2**

http://cs101.openjudge.cn/practice/21554



思路：



##### 代码

```python
# 
n=int(input())
Ti=list(map(int,input().split()))
ans1=sorted(range(1,n+1),key=lambda x:Ti[x-1])
Ti.sort()
averageTime=sum((n-i-1)*Ti[i] for i in range(n))/n

print(*ans1)
print("{:.2f}".format(averageTime))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 165054.png)



**19963:买学区房**

http://cs101.openjudge.cn/practice/19963



思路：



##### 代码

```python
# 
n=int(input())
val=[]
ans=0
pairs=[i[1:-1] for i in input().split()]
dist=[sum(map(int, i.split(","))) for i in pairs]
price=list(map(int,input().split( )))


for d in range(len(dist)):
    val.append(dist[d]/price[d]) 
    
def med(n,lis):
    if n%2==0:
        return (lis[n//2]+lis[n//2-1])/2
    else:
        return(lis[n//2])
    
val2=sorted(val)
price2=sorted(price)

med_val=med(n,val2)
med_price=med(n,price2)

for a in range(n):
    if val[a]>med_val and price[a]<med_price:
        ans +=1
        
print(ans)
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 165229.png)



**27300: 模型整理**

http://cs101.openjudge.cn/practice/27300



思路：



##### 代码

```python
# 
from collections import defaultdict
n = int(input())
d = defaultdict(list)
for _ in range(n):
    name, para = input().split('-')
    if para[-1]=='M':
        d[name].append((para, float(para[:-1])/1000) )
    else:
        d[name].append((para, float(para[:-1])))

sd = sorted(d)
for k in sd:
    paras = sorted(d[k],key=lambda x: x[1])
    value = ', '.join([i[0] for i in paras])
    print(f'{k}: {value}')
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-12 175957.png)



## 2. 学习总结和收获

题目有难度，大部分是通过学习及研究他人的代码才做出来。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





