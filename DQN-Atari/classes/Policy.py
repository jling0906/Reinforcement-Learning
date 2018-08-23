import numpy as np



class Policy:

	def getAction(self, **kwargs):
		raise NotImplementedError("This method should be overriden")

"""
random select an action 
usually being used at the beiginning of training when
the model are based random initializations
"""
class UniformRandomPolicy(Policy):

	def __init__(self, numActions):
		assert numActions >= 1
		self.numActions = numActions

	def getAction(self, qValues):
		return np.random.randint(0, self.numActions)

"""
pick up the action based the q value
dont care how many action there is, just pick up the largest
Usually being used when the model has been run for sometime and is pretty good enough
"""
class GreedyPolicy(Policy):

	def getAction(self, qValues):
		return np.argmax(qValues)

"""
Use greedy + random
when training runs for sometime, and may not be reliable
so add some randoms to search the ourside world
"""
class GreedyEpsilonPlicy(Policy):

	def __init__(self, numActions, epsilon=0.05): #TODO can make change here
		assert numActions >= 1
		self.numActions = numActions
		self.epsilon = epsilon

	def getAction(self, qValues):
		assert self.numActions == qValues.shape[1]

		if np.random.random() >= self.epsilon:
			return np.argmax(qValues)
		else:
			return np.random.randint(0, self.numActions)

"""
its the upgrade version of GreedyEpsilonPolicy
the probablity of random is big then goes now when training goes further
this policy sounds most resaonable
"""
class LinearDecayGreedyEpsilonPolicy(Policy):

	def __init__(self, numActions, initValue = 0.99, finalValue = 0.1, numSteps = 1000000):

		assert numActions >= 1
		self.numActions = numActions
		self.initValue = initValue
		self.finalValue = finalValue
		self.numSteps = numSteps
		self.steps = 0.0

	def getAction(self, qValues):


		assert self.numActions == qValues.shape[1]
		epsilon = self.initValue + (self.finalValue - self.initValue)/self.numSteps*self.steps
		
		if self.steps < self.numSteps:
			self.steps += 1.0

		if np.random.random() > epsilon:
			return np.argmax(qValues)
		else:
			return np.random.randint(0, self.numActions)




