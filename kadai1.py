import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('kadai2.dat')
average=data.mean()
std=np.std(data)

def gaussian(x,mean,sigma):
    gauss=1/np.sqrt(2.0*np.pi)/sigma*np.exp(-((x-mean)/sigma)**2/2)
    return(gauss)

def main():
    print (data)
    print('average %f' %average)
    print('standard deviation %f' %std)
    gaussian_x=sorted(data)
    y=gaussian(gaussian_x,average,std)

    plt.plot(gaussian_x,y)
    plt.hist(gaussian_x,normed=True)
    plt.show()


if __name__ == '__main__':
    main()
