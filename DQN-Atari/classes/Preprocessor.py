import numpy as np
from PIL import Image
from .ModelParams import ModelParams

"""
This class is used to preprocess data
"""
class Preprocessor:

	def __init__(self):
		params = ModelParams.getParams()
		self.imageSize = (params.inputImageSize, params.inputImageSize)
		self.numFrames = params.frameSkip
		self.reset()

	def processReward(self, reward):
		if reward > 0:
			return 1
		elif reward < 0:
			return -1
		else: 
			return 0

	# convert the images states pixels rgb values to [0,1] range
	def processBatch(self, states, nextStates):
		return states.astype("float32")/255, nextStates.astype("float32")/255

	# convert actions to one-hot encoding
	def processAction(self, actions, numActions):
		result = np.zeros((len(actions), numActions), dtype="float32")
		result[np.arange(len(actions), dtype="int"), actions] = 1.
		return result

	# process states to unint 8
	def processStateForReplayMemory(self, state):
		return self.processState(state, "L", 1)

	# process state to float 32
	def processStateForModel(self, state):
		state = self.processState(state, "F", 255.)
		if len(self.history) and self.history[0] is None:
			for i in range(self.numFrames):
				self.history[i] = state
		else:
			self.history[0: self.numFrames-1]= self.history[1: self.numFrames]
			self.history[self.numFrames - 1] = state

		return np.expand_dims(np.asarray(self.history), 0)

	# preprocess the state image
	def processState(self, state, mode, scale):
		state = Image.fromarray(state, "RGB")
		state = state.convert(mode = mode)
		shortSide = min(state.width, state.height)
		longSide = max(state.width, state.height)
		#print("**************")
		#print(self.imageSize[0])
		#print(str(int(longSide*float(self.imageSize[0]))/shortSide))
		state = state.resize((self.imageSize[0], int(longSide*float(self.imageSize[0])/shortSide)))
		state = self.cropImage(state, self.imageSize[0], self.imageSize[1])
		state = np.asarray(state)/scale

		return state

	# reset the history to None value array
	def reset(self):

		self.history = [None] * self.numFrames
	

	# crop image to new height and new width
	def cropImage(self, image, newHeight, newWidth):
		width, height = image.size

		left   = (width  - newWidth)  / 2
		right  = (width  + newWidth)  / 2
		top    = (height - newHeight) / 2
		bottom = (height + newHeight) / 2

		return image.crop((left, top, right, bottom))

