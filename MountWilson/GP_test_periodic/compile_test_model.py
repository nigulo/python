import pickle
from pystan import StanModel

model = StanModel(file="MW_test.stan")

# save it to the file 'model.pkl' for later use
with open('model_test.pkl', 'wb') as f:
    pickle.dump(model, f)

