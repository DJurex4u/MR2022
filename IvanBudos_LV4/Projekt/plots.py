import matplotlib.pyplot as plt

image1 = plt.imread('layers_3/lbfgs/10_30_5_relu_func10.png')
image2 = plt.imread('layers_3/lbfgs/10_30_5_relu_func30.png')
image3 = plt.imread('layers_3/lbfgs/10_30_5_relu_func60.png')

f, axarr = plt.subplots(1,3,figsize=(15,15))
axarr[0].imshow(image1)
axarr[1].imshow(image2)
axarr[2].imshow(image3)

plt.show()