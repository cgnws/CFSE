from multiprocessing import Pool
import os
import time


def fun(i):
    print('正在运行进程： ', i)
    time.sleep(2)


def end_fun(i=None):
    print('完成！')
    print("回调进程：:", os.getpid())


if __name__ == '__main__':
    print("主进程:", os.getpid())
    pool = Pool(3)  # 限制同时进行的进程数为 3
    for i in range(5):
        pool.apply_async(func=fun, args=(i, ), callback=end_fun)  # 并行执行，callback回调由主程序回调
        # pool.apply(func=fun, args=(i,))  # 串行执行

    # 注意：join必须放在close()后面，否则将不会等待子进程打印结束，而直接结束
    pool.close()  # 关闭进程池
    pool.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭

