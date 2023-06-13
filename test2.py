import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Create some data
x = np.linspace(0, 1, 100)
y = x ** 2

# Create a scatter plot with the "viridis" colormap
plt.scatter(x, y, c=y, cmap=cm.viridis)

# Create a line plot with the "plasma" colormap
plt.plot(x, y, c=y, cmap=cm.plasma)

plt.show()
