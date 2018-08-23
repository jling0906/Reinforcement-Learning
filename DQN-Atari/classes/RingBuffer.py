import random
from collections import deque
import numpy as np

class RingBuffer:
    
    # max size is the number of windows
    def __init__(self, maxSize, windowLength):

        psize = 84  # TODO: should be from the parameter file
        
        memSize = (maxSize + windowLength - 1);

        #print("memSize, psize, psize")
        #print("%d %d %d"%(memSize, psize, psize))
        # in total 4 circular array
        self.memState = np.ones((memSize, psize, psize), dtype=np.uint8)
        #print("memstate after creation")
        #print(self.memState)
        self.memAction = np.ones(memSize, dtype=np.int8)
        self.memReward = np.ones(memSize, dtype=np.float32)
        self.memDone = np.ones(memSize, dtype=np.bool)

    # The following are all getter and setters
    def getRingBuffer(self):
        return (self.memState, self.memAction, self.memReward, self.memDone)

    def getRingBufferByIndex(self, index):
        return (self.memState[index], self.memAction[index], self.memReward[index], self.memDone[index])

    """
    def getMemStateList(self, indexes):
        return self.memState[index]
    """

    def getMemStateList(self, indexes):
        #print("memState in getMemStateList")
        #print(self.memState)
        result = []
        #print("indexes")
        #print(indexes)
        for arr in indexes:
            result.append(self.memState[arr])
            #print("arr")
            #print(arr)
        #print("result")
        #print(result)

        return result

    def setRingBufferByIndex(self, memState, memAction, memReward, memDone, index):
        self.memState[index] = memState
        self.memAction[index] = memAction
        self.memReward[index] = memReward
        self.memDone[index] = memDone

    def setMemStateByIndex(self, memState, index):
        self.memState[index] = memState
