import numpy as np
import math
d = 2000000 * 2000000
print(d)

a  = 0.2
for i in range(300000) :
    a = a - 0.00001*a
print(a)
c = [10]
print(math.log(10))
print(np.log(np.asarray(c)))
a = [[1]]
b = [[2]]
print(a+b)