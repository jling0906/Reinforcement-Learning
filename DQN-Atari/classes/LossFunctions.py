import keras.backend as K


# seudo huber loss delta^2 * { sqrt[ 1 + (a/ delta) ^ 2] - 1}
def pseudoHuberLoss(y, pred, delta=1.):
	diff = y - pred
	diff = diff * diff
	delta = delta * delta
	result = delta * (K.sqrt(1. + diff/delta) - 1.0)
	return result


# mean huber loss
def meanPseudoHuberLoss(y, pred, delta = 1.0):
	loss = pseudoHuberLoss(y, pred, delta=delta)
	return K.mean(loss, axis = None)
