import math
import sys
from memory_profiler import profile

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

zeros = [[0]*3]*2
zeros[2][0]
print(zeros)