
import json
import numpy as np
from keras.models import load_model
import netw_models
import netw_test_predict


file_test = 'dataset_test'   # name of data file to load
model_name = 'xxx'           # model name if a whole model is imported
model_weights_name = 'u_net_model_weights'  # name of weights set if only weights are imported (usual case)
load_entire_model = False    # if False then the provided set of weights is loaded, otherwise the whole model
eval_test = True             # evaluate network without using dense crf post-processing
eval_test_CRF = True         # apply dense crf post-processing

# Load data files
with np.load(file_test + '.npz') as data:
    test_X = data['data_X']
    test_Y = data['data_Y']
    
_, height, width, channels = test_X.shape
n_classes = test_Y.shape[-1]

# Definition of various input parameters
model_args = (height, width, channels, n_classes)
average = None
dense_CRF_kwargs = {'pairwise_gauss': True, 'pairwise_bilateral': True,
                    'pw_gauss_sdims': (2, 2), 'pw_gauss_compat': 1,
                    'pw_bilat_sdims': (3, 3), 'pw_bilat_schan': (1.2,),
                    'pw_bilat_compat': 3, 'inf_steps': 10}

# Initialize model
if load_entire_model == True:
    model = load_model(model_name + '.h5')
else:
    json_name = model_weights_name.replace('weights', 'init')
    with open(json_name + '.json') as f:
        model_kwargs = json.load(f)
    model = netw_models.u_net_model(*model_args, **model_kwargs)
    model.load_weights(model_weights_name + '.h5')
    
my_model = netw_test_predict.TrainedModel(model)

# Evaluate network on test data
if eval_test == True:
    test_metrics = my_model.model_test(test_X, test_Y,
                                       verbose=1, average=average)
    print(test_metrics)
if eval_test_CRF == True:
    test_metrics_CRF = my_model.model_denseCRF_test(test_X, test_Y,
                                                    verbose=1, average=average,
                                                    **dense_CRF_kwargs)
    print(test_metrics_CRF)

del model, my_model, test_X, test_Y


