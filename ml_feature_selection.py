"""
Created on Tue Sep 18 17:48:33 2018

@author: Peng

-> This function is to compare feature importances using different methodologies
"""
from sklearn.linear_model import (LinearRegression, Ridge, 
                                  Lasso #, RandomizedLasso
								  )
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def select_features(X, Y):
    x_nm = X.columns
    ranks = {}
    
    def rank_to_dict(ranks, names, order=1):
    	minmax = MinMaxScaler()
    	ranks = minmax.fit_transform(np.array([ranks]).T).T[0]
    	ranks = map(lambda x: round(x, 2), ranks)
    	return dict(zip(names, ranks))
     
    lr = LinearRegression(normalize=True)
    ridge = Ridge(alpha=7)
    lasso = Lasso(alpha=.05)
#    rlasso = RandomizedLasso(alpha=0.04)
    #stop the search when 5 features are left (they will get equal scores)
    rfe = RFE(lr, n_features_to_select=5)
    rf = RandomForestRegressor()
    f_rg, pval  = f_regression(X, Y, center=True)
    
    dict_model = {
    				'LR': lr,
    				'Ridge': ridge,
    				'Lasso': lasso,
#    				'Stability': rlasso,
    				'RFE': rfe,
    				'RF': rf,
    				'Corr.': f_rg
    			 }
    
    for key, value in dict_model.items():
    	value.fit(X, Y)
    	if key == 'Stability':
    		ranks[key] = rank_to_dict(np.abs(value.scores_), x_nm)
    	elif key == 'RFE':
    		ranks[key] = rank_to_dict(map(float, list(value.ranking_)), x_nm, order=-1)
    	elif key == 'RF':
    		ranks[key] = rank_to_dict(value.feature_importances_, x_nm)
    	elif key == 'Corr.':
    		ranks[key] = rank_to_dict(value, x_nm)
    	else:
    		ranks[key] = rank_to_dict(np.abs(value.coef_), x_nm)

    r = {}
    for name in x_nm:
    	r[name] = round(np.mean([ranks[method][name] 
    							 for method in ranks.keys()]), 2)
     
    methods = sorted(ranks.keys())
    ranks["Mean"] = r
    methods.append("Mean")
     
    print ("\t%s" % "\t".join(methods))
    for name in x_nm:
    	print ("%s\t%s" % (name, "\t".join(map(str, 
    						 [ranks[method][name] for method in methods]))))
    return 