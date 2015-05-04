import multiprocessing as mpc
from multiprocessing import Process, Lock

def f(v, cond, n):
     with cond:
#         aux = v.value
         v.value+=1
         print(v.value)
#    v.get_lock().release()

if __name__ == '__main__':
#    lock = Lock()
    cond = mpc.Condition()
    v = mpc.Value('I', 0)
    

    for num in range(1000):
        Process(target=f, args=(v, cond, num)).start()
