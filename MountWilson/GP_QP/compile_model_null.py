import pickle
from pystan import StanModel

model_null = StanModel(file="MW_null.stan")

# save it to the file 'model.pkl' for later use
with open('model_null.pkl', 'wb') as f:
    pickle.dump(model_null, f)

