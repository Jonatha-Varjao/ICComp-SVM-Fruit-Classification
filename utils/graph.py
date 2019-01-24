from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from PIL import Image

img = Image.open('/home/varjao/Documentos/PIBICWARLEY/ICComp-SVM-Fruit-Classification/database/LaranjaInfectada/Extracted_Images/2013_10_31__0107.jpg')
red,green,blue = [],[],[]
i=0
for pixel in img.getdata():
    if i == 50:
        red.append(pixel[0])
        green.append(pixel[1])
        blue.append(pixel[2])
        i=0
    i+=1
print(max(blue))

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.asarray(red)
Y = np.asarray(green)
Z = np.asarray(blue)

# Plot the surface.
surf = ax.scatter(X, Y, Z, linewidth=0.05, 
                        c=Z, cmap='magma', marker="o")

# Customize the z axis.
ax.set_xlim(0,255)
ax.set_ylim(0,255)
ax.set_zlim(0,255)
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue')


# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=1, aspect=5)

plt.show()