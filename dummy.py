import os
import multiprocessing as mp
import time
semaphore = 0

def func1(a):
    global semaphore
    while semaphore:
        pass
    semaphore = 1
    print("a: {}".format(a))
    print("parent: {}".format(os.getppid()))
    print("pid: {}".format(os.getpid()))
    semaphore = 0
    time.sleep(2)
    return

if __name__=="__main__":
    pool = mp.Pool(processes=4,)
    pool.map(func1, [1,2,3,4,5,6])
    pool.close()
    pool.join()