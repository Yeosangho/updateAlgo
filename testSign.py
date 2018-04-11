import math
import sys
from memory_profiler import profile
import numpy as np
import resource
import gc
a = -100
b = -100

@profile
def sign(x): return 1 if x >= 0 else -1
print (math.log(100))
c = sign(a) * math.log(1+abs(a))
print(c)
print((110-100)/110)
print(math.pow(1/110, 1.5))
x=math.log(50000)/math.log(200000)
ps = math.pow(x, 0.4)
print(x)
x = 2000000 * 2000000
x= math.log(2000000)
print(x)
print(sys.getsizeof(x))

zeros = [[1,2,3],[4,5,6]]

print(zeros[1][0])
i = [0.0]*(2*250000-1)
print(resource.getrlimit(i))
for j in range(2000000) :
    i[0] += float(2000000.0)
print((i[0]))
gc.collect()