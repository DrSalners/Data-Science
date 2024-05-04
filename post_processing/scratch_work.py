from utilities.utilities import functions
import numpy as np
import matplotlib.pyplot as plt

# ### autocorr testing

# results = functions.autocorr([0.4,0.2,1.56,8.1,10,34.9,1.53,5.7], 4)
# print(type(results))

# steps, rk = functions.autocorr(np.array([0, 4, 3, 5, 3, 2, 4, 4, 6, 7, 6, 7, 2, 4, 5, 6]), 4)
# print(steps)
# print(rk)

#########################################################

# ### find_nearest testing
# ind=functions.find_nearest([4,5,2,3,24.5,6,4,56.7,-1,1,-45.6],32)
# print(ind)

# #########################################################

# ### parabdist testing
# dist=functions.parabdist(10,0.1,for_test=True)
# print(dist)

# # #########################################################

# ### getcumdist testing
# results=functions.get_cum_dist(np.array([0,1,3,4,np.nan]))
# print(type(results[0]))

### get_pdf testing
data=np.array([0,1,3,4,8,34,5.6,2,67,8,3.4,6,np.nan])
[x,y]=functions.get_pdf(data,
                    4,
                    type='lo',
                    density=True,
                    )
print(x,y)