from sklearn.base import TransformerMixin

class var_date_hour(TransformerMixin):
    
    def transform(self, X_, *_):
#         return X.InterventionDateTime.dt.hour
        X_= X_.copy()
        X_['InterventionDateTime'] = pd.to_datetime(X_.InterventionDateTime).dt.hour
#         X['InterventionDateTime'] = pd.to_datetime(X.InterventionDateTime)
#         X['InterventionDateTime'] = X.InterventionDateTime.dt.hour
        return X_
    
    def fit(self, *_):
        return self