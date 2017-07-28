import numpy as np
M = 1000000
towrite = """
# coding: utf-8

# In[1]:

import numpy as np
import cplex
from cplex.exceptions import CplexError
import os
import ntpath
import matplotlib.pyplot as plt
M = 1000000 # M


# In[2]:

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    x = int(f * 10 ** n)
    return float(x)/(10**n)


# In[3]:

def get_index_x(K, N, k, i, j):
    # k < K; i, j <= N
    return k*(N + 1)**2 + i*(N + 1) + j
def get_index_s(K, N, k, i):
    return K*(N + 1)**2 + k*(N + 1) + i


# In[4]:

def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
def distance_matrix(x, y, r):
    # Set r = precision
    size = len(x)
    t = np.array([0] * (size * size), dtype='float64')
    t = t.reshape((size,size))
    for i in range(size):
        for j in range(size):
            d = truncate(distance(x[i], y[i], x[j], y[j]),r)
            t[i,j] = d
    return t
def sign(i):
    if i > 0:
        return 1
    else:
        return 0


# N: number of customers
# K: number of vehicles
# Q: capacity
# a_i, b_i: time windows for customer i
# q_i: demand of customer i
# serv_i: service time of customer i
# t: distance matrix

# Variables: $[x, s]$  (dimension $K(N + 1)^2 + K(N + 1)$)
# $x_{ij}^k$ where $j: 0 \to N$, $i: 0 \to N$, $k: 0 \to K - 1$
# $s_i^k$ where $i: 0 \to N$, $k: 0 \to K - 1$

# In[5]:

def constraints_x(my_prob, K, N, Q, q, t): # add x-variables and contraints
    print("adding constraints_x...")
    for k in range(K):
        for i in range(N + 1):
            for j in range(N + 1):
                my_prob.variables.add(obj=[t[i,j]], ub=[1], types=['I'])


    # at least one route goes to x_i
    for i in range(1, N + 1):
        u = []
        v = []
        for j in range(N+1):
            for k in range(K):
                if i != j:
                    u.append(get_index_x(K, N, k, i, j))
                    v.append(1)
        my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "G", rhs = [sign(q[i])])

    # in = out
    for i in range(N + 1):
        for k in range(K):
            u = []
            v = []
            for j in range(N+1):
                if i != j:
                    u.append(get_index_x(K, N, k, i, j))
                    v.append(1)
                    u.append(get_index_x(K, N, k, j, i))
                    v.append(-1)
            my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "E", rhs = [0])

    # at most one vehicle each type departs from the depot
    for k in range(K):
        u = []
        v = []
        for j in range(1, N+1):
            u.append(get_index_x(K, N, k, 0, j))
            v.append(1)
        my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "L", rhs = [1])

    # capacity constraint
    for k in range(K):
        u = []
        v = []
        for i in range(N+1):
            for j in range(N+1):
                if i != j:
                    u.append(get_index_x(K, N, k, i, j))
                    v.append(q[i])
        my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "L", rhs = [Q])


# In[9]:

def constraints_s(my_prob, K, N, a, b, serv, t): # add s-variables and constraints, must be added after x-variables
    print("adding constraints_s...")
    for k in range(K):
        for i in range(N + 1):
            my_prob.variables.add(obj = [0], lb = [a[i]],ub = [b[i]],types = ['C'])
    for i in range(0, N + 1):
        for j in range(1, N+1):
            for k in range(K):
                if i != j:
                    u = [get_index_s(K, N, k, i), get_index_s(K, N, k, j), get_index_x(K, N, k, i, j)]
                    v = [1, -1, M]
                    my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "L", rhs = [M - serv[i] - t[i,j]])
    for i in range(1, N + 1):
        for k in range(K):
            u = [get_index_s(K, N, k, i), get_index_x(K, N, k, i, 0) ]
            v = [1, M]
            my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "L", rhs = [M - serv[i] - t[i,0] + b[0]])

def constraints_removed_arcs(my_prob, K, N, removed_arcs):
    print("adding contraints on removed arcs...")
    print(len(removed_arcs))
    for i in range(N+1):
        for j in range(N+1):
            if removed_arcs[i][j] != 0:
                for k in range(K):
                    u = [get_index_x(K, N, k, i, j)]
                    v = [1]
                    my_prob.linear_constraints.add(lin_expr = [[u, v]], senses = "E", rhs = [0])

# In[10]:

def read_file(file_name):
    x, y, q, a, b, serv = [], [], [], [], [], []
    K, Q = 0, 0
    with open (file_name) as f:
        lines = f.readlines()
        N = len(lines) - 10
        K,Q = lines[4].split()
        K = int(K)
        Q = int(Q)
        for i in range(9, 10 + N):
            _,x_i, y_i, q_i, a_i, b_i, serv_i  = str(lines[i]).split()
            x.append(int(x_i))
            y.append(int(y_i))
            q.append(int(q_i))
            a.append(int(a_i))
            b.append(int(b_i))
            serv.append(int(serv_i))
    t = distance_matrix(x, y, 1)
    return K, Q, x, y, q, a, b, serv, t

def read_removed_arcs(removed_arcs_file, all_loc):
    N = len(all_loc) - 1
    removed_arcs = np.zeros((N+1) * (N+1))
    removed_arcs = removed_arcs.reshape((N+1) , (N+1))
    with open(removed_arcs_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            x_1, y_1, x_2, y_2 = line.strip().split()
            loc_1 = x_1+ " " + y_1
            loc_2 = x_2+ " " + y_2
            if loc_1 in all_loc and loc_2 in all_loc:
                removed_arcs[all_loc[loc_1]][all_loc[loc_2]] = 1
    return removed_arcs

# In[17]:

def solve(file_name,removed_arcs_file): # populate by rows
    K, Q, x, y, q, a, b, serv, t = read_file(file_name)
    N = len(x) - 1
    all_loc = {}
    for i in range(N + 1):
        all_loc[str(x[i]) + " " + str(y[i])] = i
    removed_arcs = read_removed_arcs(removed_arcs_file, all_loc)
    try:
        my_prob = cplex.Cplex()
        my_prob.parameters.threads.set(1)
        my_prob.parameters.timelimit.set(%s)
        my_prob.parameters.randomseed.set(%s)
        constraints_x(my_prob, K, N, Q, q, t)
        constraints_s(my_prob, K, N, a, b, serv, t)
        constraints_removed_arcs(my_prob, K, N, removed_arcs)
        print("begin solving...")
        my_prob.solve()
        print ("Solution value  = ", round(my_prob.solution.get_objective_value(),1))
        sol = round(my_prob.solution.get_objective_value(),1)
        sol_ = my_prob.solution.get_values()
        num_vehicles = 0
        for j in range(1, N + 1):
            for k in range(K):
                num_vehicles += sol_[get_index_x(K, N, k, 0, j)]
        # plt.plot(x,y, 'o')
        # for i in range(N + 1):
        #     for j in range(N+1):
        #         for k in range(K):
        #             if sol[get_index_x(K, N, k, i, j)] > 0.5:
        #                 plt.plot([x[i], x[j]], [y[i], y[j]], 'r')
        # plt.show()
        return sol, num_vehicles
    except CplexError as exc:
        print(exc)
        return 0, 0
def write_result(file_name, sol, num_vehicles):
    with open(file_name, "w") as f:
        f.write(str(sol) + " " + str(num_vehicles))
# In[18]:
if __name__ == '__main__':
    train_set = []
    with open("%s", "r") as f:
        train_set = f.readlines()
        train_set = [ff.strip() for ff in train_set]
    c = 0
    for file in train_set:
        print(file)
        sol, num_vehicles = solve("%s"+file,"%s")
        write_result("%s" + file + "%s",sol, num_vehicles)
"""
for i in range(10,20):
    f = "cplex-Solver-" + str(i) + ".py"
    with open(f, "w") as f:
        f.write(towrite % ("3600",str(np.random.randint(1,M)), "../removed-arcs/train_rc101","../Test_in/1/",\
        "../removed-arcs/rc101_12", "../cplex-in-out/r1-rc101-" + str(i - 10) + "/", "_3600_r_12"))
        # f.write(towrite)
