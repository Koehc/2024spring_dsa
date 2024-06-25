# Assignment #2: 编程练习

Updated 0953 GMT+8 Feb 24, 2024

2024 spring, Complied by石芯洁 数学科学学院 ==同学的姓名、院系==



**说明：**

1）The complete process to learn DSA from scratch can be broken into 4 parts:
- Learn about Time and Space complexities
- Learn the basics of individual Data Structures
- Learn the basics of Algorithms
- Practice Problems on DSA

2）请把每个题目解题思路（可选），源码Python, 或者C++（已经在Codeforces/Openjudge上AC），截图（包含Accepted），填写到下面作业模版中（推荐使用 typora https://typoraio.cn ，或者用word）。AC 或者没有AC，都请标上每个题目大致花费时间。

3）课程网站是Canvas平台, https://pku.instructure.com, 学校通知3月1日导入选课名单后启用。**作业写好后，保留在自己手中，待3月1日提交。**

提交时候先提交pdf文件，再把md或者doc文件上传到右侧“作业评论”。Canvas需要有同学清晰头像、提交文件有pdf、"作业评论"区有上传的md或者doc附件。

4）如果不能在截止前提交作业，请写明原因。



**编程环境**

==（请改为同学的操作系统、编程环境等）==

操作系统：Microsoft Windows [版本 10.0.22621.2283] (c)

Python编程环境：Spyder（Python 3.11)

## 1. 题目

### 27653: Fraction类

http://cs101.openjudge.cn/practice/27653/



思路：



##### 代码

```python
# 
fractions=list(map(int,input().split()))
sim_num=[]
sim_den=[]
sim={}
if fractions[1] != fractions[3]:
    denominator=fractions[1]*fractions[3]
    numerator_1=fractions[0]*fractions[3]
    numerator_2=fractions[2]*fractions[1]
    numerator_ans=numerator_1 + numerator_2
    denominator_ans=denominator

else:
    denominator_ans=fractions[1]
    numerator_ans=fractions[0]+fractions[2]

for i in range(1,numerator_ans//2 +1):
    if numerator_ans %i==0:
        sim_num+=[i,numerator_ans/i]

for j in range(1,denominator_ans//2 +1):
    if denominator_ans %j==0:
        sim_den+=[j,denominator_ans/j]
        
for x in sim_num:
        sim[x]=1

for y in sim_den:
    if y in sim:
        sim[y]+=1

for i in sim:
    if sim[i]>1 and numerator_ans %i==0 and denominator_ans %i==0:
        numerator_ans=int(numerator_ans/i)
        denominator_ans=int(denominator_ans/i)

print(f"{numerator_ans}/{denominator_ans}")
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-10 235958.png)



### 04110: 圣诞老人的礼物-Santa Clau’s Gifts

greedy/dp, http://cs101.openjudge.cn/practice/04110



思路：



##### 代码

```python
# 
n,w_deer=map(int,input().split())
l=[]

for _ in range(n):
    v,w_candy=map(int,input().split())
    l += [v/w_candy]*(w_candy)
    
l.sort(reverse=True)
v_max=sum(l[0:w_deer])
    
print(v_max)
```



代码运行截图 ==（至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-11 000053.png)



### 18182: 打怪兽

implementation/sortings/data structures, http://cs101.openjudge.cn/practice/18182/



思路：



##### 代码

```python
# 
for _ in range(int(input())):
    n,m,b=map(int,input().split())
    d={}
    for skill in range(n):
        ti,xi=map(int,input().split())
        if ti not in d.keys():
            d[ti]=[xi]
        else:
            d[ti].append(xi)
            
    for i in d.keys():
        d[i].sort(reverse=True)
        d[i]=sum(d[i][:m])
    damage=sorted(d.items())
    for j in damage:
        b -= j[1]
        if b<=0:
            print(j[0])
            break
    else:
        print("alive")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-11 000211.png)



### 230B. T-primes

binary search/implementation/math/number theory, 1300, http://codeforces.com/problemset/problem/230/B



思路：



##### 代码

```python
# 
n=int(input())
array=list(map(int,input().split()))
max_num=max(array)
store=[True]*(int(max_num**0.5) +1)
store[0]=False
store[1]=False
 
for i in range(2,int(max_num**0.5) +1):
    if store[i]==True:
        for j in range(i*2,int(max_num**0.5)+1,i):
            store[j]=False
            
def t_prime(y):
    if y**0.5==int(y**0.5) and y>3:
        if store[int(y**0.5)]:
            return True
        else:
            return False
    return False
 
 
for _ in range(n):
    if t_prime(array[_]):
        print("YES")
    else:
        print("NO")
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-11 000307.png)



### 1364A. XXXXX

brute force/data structures/number theory/two pointers, 1200, https://codeforces.com/problemset/problem/1364/A



思路：



##### 代码

```python
# 
t=int(input())
from itertools import accumulate       #[1,2,3] accumulate得[1,3,6]
 
def prefix_sum(nums):
    return list(accumulate(nums))   
def suffix_sum(nums):
    return list(accumulate(reversed(nums)))[::-1]  #得[3,2,1],得[3,5,6],得[6,5,3]
 
for _ in range(t):
    n,x=map(int,input().split())
    array=list(map(int,input().split()))
    
    prefix=prefix_sum(array)
    suffix=suffix_sum(array)
    left=0
    right=n-1
    leftmax=-1
    rightmax=-1
    if left==right:
        if array[0]%x!=0:
            print(1)
        else:
            print(-1)
        continue
    else:
        while left < right:
            if suffix[left]%x !=0:       #从头开始删
                leftmax=n-left
                break
            else:
                left+=1
        left=0
        while right > left:
            if prefix[right]%x !=0:      #从尾开始删
                rightmax=right+1
                break
            else:
                right-=1
    
    print(max(leftmax,rightmax))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-11 000429.png)



### 18176: 2050年成绩计算

http://cs101.openjudge.cn/practice/18176/



思路：



##### 代码

```python
# 
m,n=map(int,input().split())
t=[True]*(10001)

for x in range(0,10001):
    t[0]=False
    t[1]=False
    t[2]=True
    if t[x]:
        for y in range(x*2,10000,x):
            t[y]=False
            
def t_prime(score):
    if t[int(score**0.5)]==True and int(score**0.5)==score**0.5:
        return True
    return False
            
            
for i in range(m):
    scores=list(map(int,input().split()))
    valid_scores=[score for score in scores if t_prime(score)]
    
    if len(valid_scores)==0:
        print(0)
    else:
        print("{:.2f}".format(sum(valid_scores)/ len(scores)))
```



代码运行截图 ==（AC代码截图，至少包含有"Accepted"）==

![](C:\Users\Lenovo\Pictures\Screenshots\屏幕截图 2024-03-11 000349.png)



## 2. 学习总结和收获

题目对我来说开始有难度，但在花了几个小时及研究ac代码后还是可以做出来。



==如果作业题目简单，有否额外练习题目，比如：OJ“2024spring每日选做”、CF、LeetCode、洛谷等网站题目。==





