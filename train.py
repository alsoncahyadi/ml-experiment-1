from sklearn import tree
from sklearn import neural_network
from sklearn.neural_network import MLPClassifier


class Train:
	def __init__(self, enc):
		self.enc = enc
		self.clf_nn = MLPClassifier(hidden_layer_sizes=(13,13), max_iter=200)
		self.clf_tree = tree.DecisionTreeClassifier()

		self.feature = None
		self.label = None
		self.feature_name = None
		self.encoded_feature = None
		self.encoded_label = None

	def prepare(self, data_frame):
		for datum in data_frame:
			print(datum)
		print(data_frame.head())
		self.feature = data_frame[[
			'age',
			'workclass',
			'fnlwgt',
			'education',
			'education-num',
			'marital-status',
			'occupation',
			'relationship',
			'race',
			'sex',
			'capital-gain',
			'capital-loss',
			'hours-per-week',
			'native-country'
		]].as_matrix()

		self.enc = self.enc.fit(self.feature)
		self.encoded_feature = enc.transform(self.feature)

		self.label = data_frame['income'].values
		self.encoded_label = enc.transform(self.label)

		self.feature_name = [
			'age',
			'workclass',
			'fnlwgt',
			'education',
			'education-num',
			'marital-status',
			'occupation',
			'relationship',
			'race',
			'sex',
			'capital-gain',
			'capital-loss',
			'hours-per-week',
			'native-country'
		]

	def train_dtl(self):
		self.clf_tree = self.clf_tree.fit(self.feature, self.label)
		return self.clf_tree

	def train_mlp(self):
		self.clf_nn = self.clf_nn.fit(self.feature, self.label)
		return self.clf_nn

	def get_feature(self):
		return self.feature

	def get_label(self):
		return self.label

	def get_feature_name():
		return self.feature_name
