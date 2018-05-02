import pickle
from pystan import StanModel

#model_rot = StanModel(file="MW_rot.stan")

# save it to the file 'model.pkl' for later use
#with open('model_rot.pkl', 'wb') as f:
#    pickle.dump(model_rot, f)
    

model = StanModel(file="MW.stan")

# save it to the file 'model.pkl' for later use
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

