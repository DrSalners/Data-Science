from utilities.utilities import functions

steps, rk = functions.autocorr([0, 4, 3, 5, 3, 2, 4, 4, 6, 7, 6, 7, 2, 4, 5, 6], 4)
print(steps)
print(rk)
