import numpy as np
import math
d = 2000000 * 2000000
print(d)

a  = 0.999

for i in range(1000000) :
    a = a - 0.00001*a
print(a)
c = [10]
print(math.log(10))
print(np.log(np.asarray(c)))
a = [[1]]
b = [[2]]
print(math.log(500000*2 +1,2))
#print(816*64)
print(1/math.pow(100000, 0.2))
values = [100, 200, 300, 400, 500, 600]
del values[2]
print(values)