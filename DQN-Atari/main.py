# This project is an implementation of paper "Playing Atari with Deep Reinforcement Learning"
# Project Author: Jie Ling
from classes.SpaceInvader import SpaceInvader
from classes.ModelParams  import ModelParams
from tensorflow.python.client import device_lib


# This is the entry of the project
if __name__ == "__main__":

	print("CHECK IF USING GPU")
	print(device_lib.list_local_devices())

	spaceInvader = SpaceInvader()

	args = ModelParams.getParams()
	
	if args.mode == "train":
		spaceInvader.train()
	else:
		spaceInvader.test()
