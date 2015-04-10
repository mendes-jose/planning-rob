#!/usr/bin/python
from multiprocessing import Process

class Robot(object):

    def __init__(self):
        self.name = 'bob'
        self.process = Process(target=Robot._print, args=(self,))

    def _print(self):
        print 'hello', self.name

if __name__ == '__main__':
    r1 = Robot()
    r2 = Robot()
    r1.name = 'guz'
    r1.process.start()
    r2.process.start()
    r1.process.join()
    r2.process.join()
