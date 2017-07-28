
# coding: utf-8

# In[23]:

import os
from Process_log import *

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    x = int(f * 10 ** n)
    return float(x)/(10**n)

# In[24]:

train_set = []
with open("removed-arcs/train_rc101", "r") as f:
    train_set = f.readlines()
    train_set = [ff.strip() for ff in train_set]


# In[25]:

l = len(train_set)
print(l)
best_result = [10000] * l
best_result_2 = [10000] * l
r = []
full = []
# In[26]:
def find_k(l, k):
    x = sorted(l)
    return x[k]

for f in os.listdir("cplex-in-out"):
    if f.startswith("r1-rc101"):
        file = "cplex-in-out/" + f + "/" + f + ""
        res = []
        with open(file, "r") as ff:
            res = ff.readlines()
            res = [float(x) for x in res]
        r.append(res)
    if f.startswith("full_rc101"):
        file = "cplex-in-out/" + f + "/" + f
        res = []
        with open(file, "r") as ff:
            res = ff.readlines()
            res = [float(x) for x in res]
        full.append(res)
# print(r)
r_ = list(map(list, zip(*r)))
r = [find_k(x, 8) for x in r_]
full_ = list(map(list, zip(*full)))
full = [find_k(x, 8) for x in full_]
# print(r, full)
# In[27]:

time = [0] * l
t = [0] * l
time_2 = [0] * l
t_2 = [0] * l
for j in [0,1,2,3,4,5,6,7,8,9]:
    log_file = "Subprob-cplex-Solver/out/cplex-Solver-" + str(j) + "-3600-r-4.out"
    print(log_file)
    for i in range(l):
        # print(i)
        tt = find_time(log_file, train_set[i], r[i])
        print(tt)
        if tt > 0:
            time[i] += tt
            t[i]+=1
        tt = find_time(log_file, train_set[i], full[i])
        if tt > 0:
            time_2[i] += tt
            t_2[i]+=1
print(t, t_2)
for i in range(l):
    if t[i] < 8:
        print(r[i], i)
        time[i] = 4000
    else:
        time[i] = truncate(time[i]/t[i],1)
for i in range(l):
    if t_2[i] < 8:
        time_2[i] = 4000
    else:
        time_2[i] = truncate(time_2[i]/t_2[i],1)
with open("cplex-in-out/r1_rc101", "w") as f:
    for i in range(l):
        f.write(train_set[i] + ";" + str(r[i]) + ";" + str(time[i]) +";" + str(time_2[i]) + "\n")
# print(time)
# # In[ ]:
