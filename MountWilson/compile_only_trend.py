import pickle
from pystan import StanModel

model = StanModel(file="MW_only_trend.stan")

# save it to the file 'model.pkl' for later use
with open('model_only_trend.pkl', 'wb') as f:
    pickle.dump(model, f)

