import numpy as np
from .ModelParams import ModelParams
from .RingBuffer import RingBuffer


class ReplayMemory:

	def __init__(self):
		self.args = ModelParams.getParams()
		self.maxSize = self.args.replayMemorySize
		self.windowSize = self.args.frameSkip
		self.memSize = self.maxSize + self.windowSize - 1

		self.ringBuffer = RingBuffer(self.memSize, self.windowSize)

		self.start = 0
		self.end = 0
		self.full = False

	# add one record to the Replaymemory
	def append(self, state, action, reward, done):

		# the first time append to the Replay memory
		if self.start == 0 and self.end == 0:
			for i in range(self.windowSize - 1):
				self.ringBuffer.setMemStateByIndex(state, i)
				self.start=(self.start + 1) % self.memSize

			self.ringBuffer.setRingBufferByIndex(state, action, reward, done, self.start)
			self.end = (self.start + 1)% self.memSize

		else:
			self.ringBuffer.setRingBufferByIndex(state, action, reward, done, self.end)
			self.end = (self.end + 1) % self.memSize

			# the replay memory is full
			if self.end > 0 and self.end < self.start:
				self.full = True
			if self.full:
				self.start = (self.start + 1) % self.memSize

	# pick up samples from the replay memory for training
	def sample(self):
		batchSize = self.args.batchSize

		#print("self.start is %d" % self.start)
		#print("self.end is %d" % self.end)

		# calculate the number of records in the replaymemory
		if self.end == 0 and self.start == 0: # look at end first for effciency
			return None, None, None, None
		else: 
			count = 0
			if self.end > self.start:
				count = self.end - self.start
			else:
				count = self.maxSize

			# generate the random indexes for sampling
			#print("count is %d"%count)
			#print("batchSize is %d"%batchSize)
			if count <= batchSize:
				indexes = np.arange(0, count - 1) # TODO: shoule be (0, count) when count == 1
			else:
				indexes = np.random.randint(0, count - 1, size = batchSize)
			#print("****indexes is***1**")
			#print(indexes)
			index4 = (self.start + indexes) % self.memSize # changed here

			indexes = [((np.asarray(indexes) + self.start + i) % self.memSize) for i in range(1, -4, -1)] # changed here
			#print("****indexes is***2**")
			#print(indexes)

			memstateList = self.ringBuffer.getMemStateList(indexes)

			#print("memstateList")
			#print(memstateList)

			stateList = np.transpose(memstateList[0:4], [1,0, 2, 3])
			#print("stateList")
			#print(stateList)

			nextStateList = np.transpose(memstateList[1:5], [1, 0, 2, 3])
			#print("nextStateList")
			#print(nextStateList)
			_, actionList, rewardList, doneList = self.ringBuffer.getRingBufferByIndex(index4)
			return stateList, actionList, rewardList, nextStateList, doneList

	# clear the replay memory
	def reset(self):
		#no need to set value, just set the start and end, pretending to be empty
		self.start, self.end, self.full = 0, 0, False