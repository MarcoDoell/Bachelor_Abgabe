from sklearn.neighbors import LocalOutlierFactor

def do_lof(X):
    clf = LocalOutlierFactor(n_neighbors=5)
    
    pred = clf.fit_predict(X)
    scores = clf.negative_outlier_factor_

    return pred, scores

from sklearn.svm import OneClassSVM

def do_one_class_svm(X):
    clf = OneClassSVM(gamma=0.00001, nu=0.01).fit(X)

    pred = clf.predict(X)
    scores = clf.score_samples(X)

    return pred,scores

from sklearn.ensemble import IsolationForest

def do_iForest(X):
    clf = IsolationForest(n_estimators=50, max_samples='auto', contamination=float(0.1),max_features=1.0).fit(X)
    
    pred = clf.predict(X)
    scores = clf.decision_function(X)

    return pred,scores



df_model = pd.DataFrame(df.values, columns = ['Anwender','Saldo'])
df_model['scores'] = scores
df_model['anomaly'] = pred