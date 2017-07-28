import os
import sys

arg = sys.argv[1]
# print(sys.argv)
i = 0
for file in os.listdir(arg):
	i += 1
print(i)	
