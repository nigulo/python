import pickle
from pystan import StanModel

model = StanModel(file="MW_test_wo_length_scale.stan")

# save it to the file 'model.pkl' for later use
with open('model_test_wo_length_scale.pkl', 'wb') as f:
    pickle.dump(model, f)

