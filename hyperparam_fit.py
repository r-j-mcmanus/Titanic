from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

global_scoring = 'roc_auc'

def knn_best_fit(X_train, y_train, plot = True):
	knn = KNeighborsClassifier()
	param_range = list(range(3,40))

	param_grid = {
		'n_neighbors' : param_range
			}
	knn_GS = GridSearchCV(knn, param_grid = param_grid, cv = 5, n_jobs = -1, scoring = global_scoring)
	knn_best_fit = knn_GS.fit(X_train, y_train)

	print('Best Scoring ' + global_scoring,'\nBestScore:',
	 knn_best_fit.best_score_, 
	 '\nbest params',
	  knn_best_fit.best_params_)


	plt.plot(param_range, knn_best_fit.cv_results_['mean_test_score'])
	plt.show()

	return knn_best_fit.best_estimator_


def tree_best_fit(X_train, y_train, plot = True):
	tree = DecisionTreeClassifier()
	param_range = list(range(1,20))
	param_grid = {
		'max_depth' : param_range
			}
	tree_GS = GridSearchCV(tree, param_grid = param_grid, cv = 5, n_jobs = -1, scoring = global_scoring)
	tree_best_fit = tree_GS.fit(X_train, y_train)

	print('Best Scoring ' + global_scoring,'\nBestScore:',
	 tree_best_fit.best_score_, 
	 '\nbest params',
	  tree_best_fit.best_params_)


	plt.plot(list(range(1,20)), tree_best_fit.cv_results_['mean_test_score'])
	plt.show()

	return tree_best_fit.best_estimator_


def LogisticRegression_best_fit(X_train, y_train):
	logreg = LogisticRegression() 
	param_grid = {'penalty' : ['none', 'l2'], 'max_iter': [1000]}
	logreg_GS = GridSearchCV(logreg, param_grid = param_grid, cv = 5, n_jobs = -1, scoring = global_scoring)

	logreg_best_fit = logreg_GS.fit(X_train, y_train)

	print('Best Scoring ' + global_scoring,'\nBestScore:',
	 logreg_best_fit.best_score_, 
	 '\nbest params',
	  logreg_best_fit.best_params_)

	return logreg_best_fit.best_estimator_

def SVC_best_fit(X_train, y_train):
	svc = SVC(probability=True) 
	param_grid = {'gamma' : ['scale', 'auto'], 'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
	gs = GridSearchCV(svc, param_grid = param_grid, cv = 5, n_jobs = -1, scoring = global_scoring)

	svc_best_fit = gs.fit(X_train, y_train)

	print('SVC best_fit:\n BestScore:',
	 svc_best_fit.best_score_, 
	 '\nbest params',
	  svc_best_fit.best_params_)

	return svc_best_fit.best_estimator_

####



def knn_n_neighbors_best_fit_alt(X_train, y_train, plot = True):
	k_scores = []
	k_range = list(range(5,100))

	for k in k_range:
		knn = KNeighborsClassifier(n_neighbors = k) 
		pipeline_knn = make_pipeline(knn)
		loss = abs(cross_val_score(
			pipeline_knn, X_train, y_train, cv = 5, scoring='neg_mean_squared_error'
			))
		k_scores.append(loss.mean())

	if plot == True:
		plt.plot(k_range, k_scores)
		plt.xlabel('Value of K for KNN')
		plt.ylabel('Cross-Validated MSE')
		plt.show()

	k_dict = dict(zip(k_scores, k_range))
	n_neighbors_best_fit = k_dict[min(k_scores)]

	return n_neighbors_best_fit

def DecisionTreeClassifier_depth_best_fit_alt(X_train, y_train, plot = True):
	depth_scores = []
	depth_range = list(range(1,20))

	for depth in depth_range:
		tree = DecisionTreeClassifier(max_depth = depth) 
		pipeline_tree = make_pipeline(tree)
		loss = abs(cross_val_score(
			pipeline_tree, X_train, y_train, cv = 5, scoring='neg_mean_squared_error'
			))
		depth_scores.append(loss.mean())

	if plot == True:
		plt.plot(depth_range, depth_scores)
		plt.xlabel('Value of depth for DecisionTreeClassifier')
		plt.ylabel('Cross-Validated')
		plt.show()

	depth_dict = dict(zip(depth_scores, depth_range))
	depth_best_fit = depth_dict[min(depth_scores)]

	return depth_best_fit