# This class is the core of the project
# containing train and test method

import os
import sys

import numpy as np

import gym
from gym import wrappers

import keras.models
from keras.layers import (Input, Conv2D, Dense, Flatten)
from keras.layers.merge import dot
import keras.backend as K
from keras.optimizers import Adam

import tensorflow as tf

from .ModelParams import ModelParams
from .Policy import Policy, UniformRandomPolicy, LinearDecayGreedyEpsilonPolicy, GreedyPolicy
from .Preprocessor import Preprocessor
from .ReplayMemory import ReplayMemory
from .LossFunctions import meanPseudoHuberLoss

class SpaceInvader:

	def __init__(self):

		self.args = ModelParams.getParams()

		# env related
		self.env = gym.make(self.args.env)
		self.env.reset()
		self.numActions = self.env.action_space.n

		self.policy = {
			"init":  UniformRandomPolicy(self.numActions),
			"train": LinearDecayGreedyEpsilonPolicy(self.numActions),   # TODO: can make change here
			"test":  GreedyPolicy()
		}

		self.createDQNModel()
		self.getUtils("init")
 
	# create the deep Q network model and its target model, which should be a delayed copy of the dqn model
	def createDQNModel(self):
		# Create the network model
		self.model, self.qValueFunction = self.createModel()

		# create the model the same as the network model, which would be delayed and updated during training
		self.delayedModel, self.delayedQValueFunction = self.copyModel(self.model)

	# copy the deep q network model
	def copyModel(self, model):
		delayedModel = keras.models.clone_model(model)
		delayedModel.set_weights(model.get_weights())

		delayedQValueFunction = K.function([model.layers[0].input], [model.layers[5].output])
		return delayedModel, delayedQValueFunction

	# create the deep q network model with Keras library, which is actually a wrapped tensorflow API 
	def createModel(self):
		if self.args.modelType == "DQN":
			with tf.name_scope("DQN"):
				with tf.name_scope("input"):
					inputState = Input(shape=(self.args.frameSkip, self.args.inputImageSize, self.args.inputImageSize))
					inputAction = Input(shape=(self.numActions,))

				with tf.name_scope("conv1"):
					conv1 = Conv2D(16, (8, 8), data_format = "channels_first", kernel_initializer="glorot_uniform", 
						activation="relu", padding="valid", strides=(4, 4))(inputState)

				with tf.name_scope("conv2"):
					conv2 = Conv2D(32, (4, 4), data_format="channels_first", kernel_initializer="glorot_uniform",
						activation="relu", padding="valid", strides=(2, 2))(conv1)

				with tf.name_scope("fc"):
					flattened = Flatten()(conv2)
					dense1 = Dense(256, kernel_initializer="glorot_uniform", activation="relu")(flattened)

				with tf.name_scope("output"):
					qValues = Dense(self.numActions, kernel_initializer="glorot_uniform", activation = None)(dense1)
					qV = dot([qValues, inputAction], axes = 1)

				model = keras.models.Model(inputs = [inputState, inputAction], output = qV)
				qValueFunction = K.function([inputState], [qValues])

			model.summary()

		elif self.args.modelType == "doubleDQN": # TODO
			pass
		elif self.args.modelType == "duelingDQN":
			pass
		else:
			print("Not implemented")
			sys.exit(0)

		return model, qValueFunction

	# The training process of this algorithm
	#1先随机生成一个初始状态
    #2model已经有了 权重也是随机的 model有一个初始的q function
    #3根据这个初始的qfunction 走第一步
    #4将这一步的信息放（state， action，reward， done） 放入历史记录
    #5.重复34 直到memory满了为止
    #6.memory满了以后 继续走34，memory里旧的历史删去
    #7.如果循环达到更新的频率，进行模型的权重更新：
    #--从memory里抽样
    #--对样本进行处理
    #--送入定义好的keras的神经网络networkModel
    #--更新权重
    #8.更新状态（就是刚才算出来的4里面的state）
	def train(self):
		sys.stdout.flush()

		self.args.outputDir = self.getOutputPath()
		self.compile(optimizer=Adam(lr=self.args.learningRate), lossFunction=meanPseudoHuberLoss)  # TODO: can change here
		self.getUtils("train")

		numUpdates = 0
		numEpisodes = 0

		while numEpisodes < self.args.maxNumEpisodes : # TODO: not sure here should be number of updates or episodes
			numEpisodes += 1

			self.numStepsPerGame = 0
			totalRewardsPerGame = 0

			# create a initial state which would be random
			state = self.env.reset()
			self.preprocessor.reset()

			#print("Game episode " + str(numEpisodes) + "Starts:")

			while True:
				self.numStepsPerGame += 1
				self.totalSteps += 1

				action, _ = self.getAction(state) # get the action from the policy defined by the q function

				nextState, reward, done, _ = self.env.step(action) # convert the reward to [-1, 1]

				reward = self.preprocessor.processReward(reward) 
				totalRewardsPerGame += reward

				state = self.preprocessor.processStateForReplayMemory(state)

				self.replayMemory.append(state, action, reward, done)

				# check if the memory burn-in is done

				if self.totalSteps > self.args.numBurnIn:
					if self.mode != "train":
						self.mode = "train"
						print("Finishing burning in the replay memory") # TODO: do we need to reset the memory every episode?

					if self.numStepsPerGame % self.args.modelUpdatePeriod == 0:
						self.fit()  # this is the real training part, all the above is actually doing preparations
						numUpdates += 1

						if numUpdates % 10000 == 0:  # TODO: why h5
							# model shoule be saved under modelWeights/trainingi/modelWeightsj.h5
							modelPath = "%s/modelWeights%d.h5" % (self.args.outputDir, numUpdates // 10000)
							self.model.save_weights(modelPath) # TODO: why reward is not increasing? so this weight is not the best?
							#TODO: how to update args.modelPath for testing
				if done or (self.args.maxNumEpisodes != None and self.numStepsPerGame > self.args.maxNumEpisodes):
					break

				state = nextState

			print("Episode %d is done, which uses %d steps, did %d updates on cnn and gets %d reward" % (numEpisodes, self.numStepsPerGame, numUpdates, totalRewardsPerGame))

		print("Training done *^-^*")

	# The testing process of this algorithm
	def test(self):
		if not os.path.isfile(self.args.modelPath):
			print("Model path: {} does not exist".format(self.args.modelPath))
			return

		rewards = []
		numStepsPerGameList = []
		numEpisodes = 0

		while True:

			# utilities ready
			self.getUtils("test")

			# model ready
			self.loadWeightsFromFile()

			numEpisodes += 1
			totalRewardsPerGame = 0
			self.numStepsPerGame = 0
			# environment ready
			env = gym.make(self.args.env)
			# create a video and save it in folder "videos"
			env = wrappers.Monitor(env, "videos", force = True)
			# create a random initial state
			state = env.reset()

			while True:
				self.numStepsPerGame += 1


				action, _ = self.getAction(state)
				nextState, reward, done, _ = env.step(action)
				reward = self.preprocessor.processReward(reward)
				totalRewardsPerGame += reward

				if done or (self.args.maxNumEpisodes is not None and numEpisodes > self.args.maxNumEpisodes):
					break

				state = nextState
			rewards.append(totalRewardsPerGame)
			numStepsPerGameList.append(self.numStepsPerGame)

			print("Reward of this game is %d. the number of steps is %d" % (totalRewardsPerGame, self.numStepsPerGame))
			if numEpisodes > 100 or reward > 350:  # TODO: could change here. 100 should be from args
				break 

		print(" The average reward is %d. The std of reward is %f. The average number of steps of each episode is %d"%
			(np.mean(totalRewardsPerGame), np.std(totalRewardsPerGame), np.mean(self.numStepsPerGame)))



	#  get the model saving root folder from args.outputDir and update it with complete path
	def getOutputPath(self):
		parentDir = self.args.outputDir  # TODO: No need to worry about this arg been updated to full path for now
										 # as each time we run a single operation, train or test
		os.makedirs(parentDir, exist_ok=True)

		envName = self.args.env
		subDirId = 0

		for item in os.listdir(parentDir):
			if not os.path.isdir(os.path.join(parentDir, item)):
				continue
			try:
				item = int(item.split("run")[-1])
				if item > subDirId:
					subDirId = item
			except: 
				pass
		subDirId += 1

		finalPath = os.path.join(parentDir, envName)
		finalPath += "run{}".format(subDirId)

		os.makedirs(finalPath, exist_ok=True)

		return finalPath

	# configure the optimizer and lossfunction of both the dqn model and its delayed mirror model
	def compile(self, optimizer, lossFunction):
		self.model.compile(optimizer=optimizer, loss = lossFunction)
		self.delayedModel.compile(optimizer=optimizer, loss = lossFunction)

	# configure the mode, as well as the preprocessorm and replay memory based on the status of the mode
	def getUtils(self, mode): # test
		self.totalSteps = 0 # used to update the delayed network
		if mode == "init" or mode == "test":
			self.mode = mode
			self.preprocessor = Preprocessor()
			self.replayMemory = ReplayMemory()
		elif mode == "train":
			self.mode = "init"
			self.preprocessor.reset()
			self.replayMemory.reset()

	# select the action based on the policy and q value function
	def getAction(self, state):

		state = self.preprocessor.processStateForModel(state)

		qValues = self.qValueFunction([state])[0]

		return self.policy[self.mode].getAction(qValues), state

	"""
	read the model weights from local directory, and
	assign them to the dqn model and its mirror model
	"""
	def loadWeightsFromFile(self):

		weightsPath = self.args.modelPath
		self.model.load_weights(weightsPath)
		self.delayedModel.set_weights(self.model.get_weights())

	"""
	this is the real training part, which means all the code above this function call is 
	the preparations.
	This function is to train the model and update the mirror model with Keras train_on_batch API
	"""

	def fit(self):

		# step1: sample from memory
		states, actions, rewards, nextStates, done = self.replayMemory.sample()
		#print("state")
		#print(states)
		states, nextStates = self.preprocessor.processBatch(states, nextStates)
		#print("state")
		#print(states)
		actions = self.preprocessor.processAction(actions, self.numActions)

		delayedQValues = self.delayedQValueFunction([nextStates])[0]

		maxDelayedQValue = np.max(delayedQValues, axis = 1)
		maxDelayedQValue[done] = 0.0
		targets = rewards + self.args.gamma * maxDelayedQValue
		targets = np.expand_dims(targets, axis = 1) # TODO: why need to expand the array

		#print("state")
		#print(states)
		#print("actions")
		#print(actions)
		#print("targets")
		#print(targets)
		self.model.train_on_batch([states, actions], targets)

		if self.totalSteps % self.args.delayedModelUpdatePeriod == 0:
			self.delayedModel.set_weights(self.model.get_weights())
			#print("Update the delayed Q network at step{}".format(self.numStepsPerGame))







