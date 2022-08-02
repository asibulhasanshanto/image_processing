import numpy as np

def main():
    kernel =255- np.ones((3,3),dtype=np.int8)*10
    print(kernel,kernel.dtype)

    kernel2=np.zeros((3,3),dtype=np.int8)
    print(kernel2,kernel2.dtype)

    kernel3 = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=np.int8)
    print(kernel3,kernel3.dtype,kernel3.size)

if __name__ == '__main__':
    main()