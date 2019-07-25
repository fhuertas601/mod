from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def svm(cv,x_tr,x_ts,y_tr,y_ts,nproc,verb):
# Calculation of Support Vector Machine model (SVM)
# Announce SVM model is being calculated
	print('\n  ***********')
	print('   SVM model')
	print('  ***********')
# Create grid search dictionary. C and gamma grid is really trimmable to
# search for the best parameters. Shrinking the grid too much might cause
# the SVM to fail in finding the vectors
	param_grid = {"C": [10 ** i for i in range(-6, 5)],
              "gamma": [10 ** i for i in range(-12, 2)]}
# Setup model building
	m = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, n_jobs=nproc, cv=cv, verbose=1)
# Build the model
	print('-> Fitting the data')
	model=m.fit(x_tr, y_tr)
	if verb:
		print(model)
		print('SVM best params')
		print(m.best_params_)
		print('SVM best score')
		print(m.best_score_)
		print('cv_results')
		print(m.cv_results_)
		print('mean_test_score')
		print(m.cv_results_['mean_test_score'])
		print('params')
		print(m.cv_results_['params'])
	
	return m

def rf(cv,x_tr,x_ts,y_tr,y_ts,nproc,verb):
# Calculation of Random Forest model (RF)
# Announce RF model is being calculated
    print('\n  **********')
    print('   RF model')
    print('  **********')
# create grid search dictionary 
    param_grid = {"max_features": [x_tr.shape[1] // 10, x_tr.shape[1] // 7, x_tr.shape[1] // 5, x_tr.shape[1] // 3],
            "n_estimators": [100, 250, 500]}
    if verb:
        print(param_grid)
# setup model building 
    m = GridSearchCV(RandomForestClassifier(), param_grid, n_jobs=nproc, cv=cv, verbose=1)
# run model building 
    print('\n-> Fitting the data')
    model=m.fit(x_tr, y_tr)
    if verb:
        print(model)
        print('best_params')
        print(m.best_params_)
        print('best_score')
        print(m.best_score_)
        print('cv_results')
        print(m.cv_results_)
        print('mean_test_score')
        print(m.cv_results_['mean_test_score'])
        print('params')
        print(m.cv_results_['params'])
    print('\n-> Checking the quality of the model: prediction of the test set')

    return m

def gbm(cv,x_tr,x_ts,y_tr,y_ts,nproc,verb):
# Calculation of Gradient Boost model (GBM)
# Announce RF model is being calculated
	print('\n  **********')
	print('   GBM model')
	print('  **********')
# create grid search dictionary
	param_grid = {"n_estimators": [100, 200, 300, 400, 500]}
#    if verb:
#        print(param_grid)
# setup model building 
	m = GridSearchCV(GradientBoostingClassifier(subsample=0.5, max_features=0.5), 
					 param_grid, n_jobs=2, cv=cv, verbose=1)
# run model building 
	print('\n-> Fitting the data')
	model=m.fit(x_tr, y_tr)
	if verb:
		print(model)
		print('best_params')
		print(m.best_params_)
		print('best_score')
		print(m.best_score_)
		print('cv_results')
		print(m.cv_results_)
		print('mean_test_score')
		print(m.cv_results_['mean_test_score'])
		print('params')
		print(m.cv_results_['params'])
		print('\n-> Checking the quality of the model: prediction of the test set')
	return m
