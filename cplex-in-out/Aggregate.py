import os
train_set = []
with open("../removed-arcs/train_rc101", "r") as f:
    train_set = f.readlines()
    train_set = [ff.strip() for ff in train_set]
for f in os.listdir():
    if os.path.isdir("./" + f):
        s = []
        for file in train_set:
            # if f.startswith("full_rc101"):
            #     ff = file + "_full"
            #     with open("./" + f + "/" + ff, "r") as fff:
            #         ob = fff.readlines()[0].split()[0]
            #         s.append(ob)
            if f.startswith("r1-rc101"):
                ff = file + "_3600_r"
                with open("./" + f + "/" + ff, "r") as fff:
                    ob = fff.readlines()[0].split()[0]
                    s.append(ob)
            # elif f.startswith("r-rc101"):
            #     ff = file + "_3600_r"
            #     with open("./" + f + "/" + ff, "r") as fff:
            #         ob = fff.readlines()[0].split()[0]
            #         s.append(ob)

        if f.startswith("r1-rc101"):
            with open("./" + f + "/" + f + "", "w") as ffff:
                for x in s:
                    ffff.write(x + "\n")
