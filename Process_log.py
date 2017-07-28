
# coding: utf-8

# In[53]:

def truncate(f, n):
    '''Truncates/pads a float f to n decimal places without rounding'''
    x = int(f * 10 ** n)
    return float(x)/(10**n)
def extract_log(log_file, file_name):
    log = []
    with open(log_file, "r") as l:
        lines = l.readlines()
        i = 0
        while lines[i].strip() != file_name:
            i += 1
        j = i
        while not (lines[j].startswith("Solution value") or lines[j].startswith("CPLEX Error")):
            j += 1
    k = i
    e = 0
    while k <= j:
        if lines[k].startswith("Elapsed"):
            e = float(lines[k].split()[3])
        elif len(lines[k].split()) == 9 and lines[k].startswith("*") and e > 0:
            # print(lines[k])
            e += 1
            log.append([e, truncate(float(lines[k].split()[5]), 1)])
        elif len(lines[k].split()) == 8 and e > 0:
            e += 1
            log.append([e, truncate(float(lines[k].split()[4]), 1)])
            # print([e, truncate(float(lines[k].split()[4]), 1)])
        elif len(lines[k].split()) == 7 and lines[k].split()[2] == "cutoff" and e > 0:
            e += 1
            log.append([e, truncate(float(lines[k].split()[3]), 1)])
            # print("here", [e, truncate(float(lines[k].split()[3]), 1)])
        k += 1
    return log
def find_time(log_file, file_name,t):
    log = extract_log(log_file, file_name)
    # print(log)
    # if file_name == "rc101_96":
    #     print(log)
    log_ob = [a[1] for a in log]
    index = [x[0] for x in enumerate(log_ob) if x[1] -0.1 <= t]
    # print(index)
    if len(index) == 0:
        return -1
    else:
        return log[index[0]][0]


# In[54]:

# print(find_time("Subprob-cplex-Solver/out/cplex-Solver-20-3600-r-4.out", "rc101_8", 761.8))


# In[ ]:
