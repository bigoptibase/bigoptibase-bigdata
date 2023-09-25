import pandas as pd
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

if __name__ == "__main__":
    data = pd.read_csv("data/cosmote.csv")
    seq = data.NORM_AVG_DL_MAC_CELL_TPUT
    # choose a number of time steps
    n_steps_in, n_steps_out = 7, 7
    # split into samples
    X, y = split_sequence(seq, n_steps_in, n_steps_out)