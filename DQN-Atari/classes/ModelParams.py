# This class is to contain the parameters of DQN model
import argparse

class ModelParams:

	parser = argparse.ArgumentParser(description="parser for dqn model parameters")
    

    #  model external params 
	parser.add_argument("--env", default="SpaceInvaders-v0", help="Model environment name")
	parser.add_argument("--mode", default="train", help="The mode of program.Could be train or test")
	parser.add_argument("--modelType", default="DQN", type = str, help="The type of the model, could be DQN or DDQN or DuDQN")

	parser.add_argument("--outputDir", default="modelWeights", help="The output folder of model weights. Used for training output")
	parser.add_argument("--modelPath", default="/Users/jieling/Documents/AI/Projects/Reinforcement Learning/DQN-Atari/modelWeights/SpaceInvaders-v0run52/modelWeights1.h5", help="The path where the model weights are stored. Used for testing input")

	# preprpcessor
	parser.add_argument("--frameSkip", default=4, type = int, help="The number of frames for each state")
	parser.add_argument("--inputImageSize", default=84, type=int, help="The final size of the input state image")
	
	# Replay Memory
	parser.add_argument("--replayMemorySize", default=1000000, type=int, help="The size of replay memory")
	parser.add_argument("--numBurnIn", default=12000, type=int, help="The number of frames to burn into replay memory before training")

	# Training
	parser.add_argument("--batchSize", default=32, type=int, help="The number of states of each replay memory sampling, which is also the size of each training batch")
	parser.add_argument("--learningRate", default=0.00025, type=float, help="The value of learning rate")
	parser.add_argument("--modelUpdatePeriod", default=10, type=int, help="How often to update the deep q network")
	parser.add_argument("--delayedModelUpdatePeriod", default=10000, type=int, help="How often to update the target network")

	parser.add_argument("--maxNumUpdates", default=1000, type=int, help="The maximun number of updating the model weights")
	parser.add_argument("--maxNumEpisodes", default=10000, type=int, help="The maximum number of episodes to run the entire game dring training")
	
	parser.add_argument("--gamma", default=0.99, type=float, help="The reward discount factor")
	parser.add_argument("--randomSeed", default=0, type=int, help="The random seed")


	# Return the argument parser to retrieve the algorithm parameters
	@classmethod
	def getParams(cls):
	 	return cls.parser.parse_args()