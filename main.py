import read_dataset
from train import Train
from sklearn.preprocessing import OneHotEncoder

if __name__ == '__main__':
	enc = OneHotEncoder()
	t = Train(enc)
	df = read_dataset.get_data_frame()
	t.prepare(df)
	# t.train_dtl()
	t.train_mlp()