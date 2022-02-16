# Test stacking on the sonar dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


# Split a dataset into k folds
def cross_validation_split(dataset, fold_num):
	split_dt = list()
	dt = list(dataset)
	# size_fold = int(len(dataset) / n_folds)
	for i in range(fold_num):
		folds = list()
		while len(folds) < (int(len(dataset) / fold_num)):
			folds.append(dt.pop(randrange(len(dt))))
		split_dt.append(folds)
	return split_dt

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	length=len(actual)
	for i in range(length):
		if actual[i] == predicted[i]:
			correct += 1
	return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def eva_alg(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	list_score = list()
	for fold in folds:
		train = list(folds)
		train.remove(fold)
		train = sum(train, [])
		test = list()
		for row in fold:
			m_rowlist = list(row)
			test.append(m_rowlist)
			m_rowlist[-1] = None
		predicted = algorithm(train, test, *args)
		actual = [row[-1] for row in fold]
		acc = accuracy_metric(actual, predicted)
		list_score.append(acc)
	return list_score

# Calculate the Euclidean distance between two vectors
def euclidean_distance(point1, point2):
	dist = 0.0
	for i in range(len(point1)-1):
		dist += (point1[i] - point2[i])**2
	distance=sqrt(dist)
	return distance

# Locate neighbors for a new row
def get_neigh(train, test, n_neigh):
	dist_list = list()
	for i in train:
		dist = euclidean_distance(test, i)
		dist_list.append((i, dist))
	dist_list.sort(key=lambda tup: tup[1])
	neigh = list()
	for i in range(n_neigh):
		neigh.append(dist_list[i][0])
	return neigh

# Make a prediction with kNN
def knn_predict(alg, test, num_neighbors=5):
	neighbors = get_neigh(alg, test, num_neighbors)
	out_val = [row[-1] for row in neighbors]
	predictions = max(set(out_val), key=out_val.count)
	return predictions

# Prepare the kNN model
def knn_train(train):
	return train

# Make a prediction with weights
def perceptron_predict(alg, row):
	activ_fun = alg[0]
	for i in range(len(row)-1):
		activ_fun += alg[i + 1] * row[i]
	if activ_fun>=0.0 :
		return 1.0
	else :
		return 0.0

# Estimate Perceptron weights using stochastic gradient descent
def perceptron_train(train,l_rate=0.01, n_epoch=5000):
	w = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			predicted = perceptron_predict(w, row)
			error = row[-1]-predicted
			w[0] = w[0]+l_rate*error
			for i in range(len(row)-1):
				w[i+1]=w[i+1]+l_rate*error*row[i]
	return w

# Make a prediction with coefficients
def LR_predict(alg, row):
	ycap = alg[0]
	for i in range(len(row)-1):
		ycap += alg[i + 1] * row[i]
	func=1.0 / (1.0 + exp(-ycap))
	return func

# Estimate logistic regression coefficients using stochastic gradient descent
def LR_train(train, l_rate=0.01, n_epoch=2000):
	coeffs = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			ycap = LR_predict(coeffs, row)
			error = row[-1] - ycap
			coeffs[0] = coeffs[0] + l_rate * error * ycap * (1.0 - ycap)
			for i in range(len(row)-1):
				coeffs[i+1]=coeffs[i + 1] + l_rate * error * ycap * (1.0 - ycap) * row[i]

	return coeffs

# Make predictions with sub-models and construct a new stacked row
def combine_row(algs,predict_list,row):
	row_stack = list()
	for i in range(len(algs)):
		predictions = predict_list[i](algs[i], row)
		row_stack.append(predictions)
	row_stack.append(row[-1])
	fin=row[0:len(row)-1]+row_stack
	return fin

def combine(train, test):
	alg_list = [knn_train, perceptron_train]
	predict_list = [knn_predict, perceptron_predict]
	algs = list()
	for i in range(len(alg_list)):
		alg = alg_list[i](train)
		algs.append(alg)
	s_dt = list()
	for row in train:
		s_row = combine_row(algs, predict_list, row)
		s_dt.append(s_row)
	s_alg = LR_train(s_dt)
	predicts = list()
	for row in test:
		s_row = combine_row(algs,predict_list,row)
		s_dt.append(s_row)
		predicts.append(round(LR_predict(s_alg,s_row)))
	return predicts

seed(1)
filename = 'sonar.all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	for rows in dataset:
		rows[i] = float(rows[i].strip())

class_val = [row[len(dataset[0])-1] for row in dataset]
temp = dict()
for i, value in enumerate(set(class_val)):
	temp[value] = i
for row in dataset:
	row[len(dataset[0])-1] = temp[row[len(dataset[0])-1]]
folds_num=3
scores=eva_alg(dataset, combine,folds_num)
print("scores= %s"% scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
