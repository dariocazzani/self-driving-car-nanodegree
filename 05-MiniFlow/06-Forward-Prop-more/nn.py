"""
No need to change anything here!

If all goes well, this should work after you
modify the Add class in miniflow.py.
"""

from miniflow import *

x, y, z = Input(), Input(), Input()

f = Add(x, y, z)
m = Mul(x, y, z)

feed_dict = {x: 4, y: 5, z: 10}

graph = topological_sort(feed_dict)
output_sum = forward_pass(f, graph)
output_mul = forward_pass(m, graph)

# should output 19
print("{} + {} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output_sum))

print("{} * {} * {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], feed_dict[z], output_mul))
