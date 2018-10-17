import pickle
from pystan import StanModel

model = StanModel(file="model.stan")

# save it to the file 'model.pkl' for later use
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

