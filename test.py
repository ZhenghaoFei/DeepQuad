import numpy as np
a = np.asarray([1,2,2,3,4])
print a
a = a.reshape([1, 5])
print a
a = np.squeeze(a)
print a