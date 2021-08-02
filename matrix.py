import time
import random

A = time.time()

for i in range(10000000):
	s = random.randint(0, 2**100)
	s =  '{0:08b}'.format(s)
	print(s)

B = time.time()

print(B-A)
