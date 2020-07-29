
import json
import numpy as np
from keras.models import load_model
import helper_func
import netw_models
import netw_test_predict


file_pred = 'dataset_test'
n_classes = 4
class_names = ["'boundary'", "'interior 1'", "'interior 2'", "'background'"]
model_name = 'xxx'
model_weights_name = '..._weights'
load_entire_model = False
predict_all = True
single_file_number = 15
plot_prediction = True
plot_mask_with_input = True
input_to_plot = 2
input_name = r'Phase image'
class_to_plot = (2, 3)
use_test_set = True
xtick_int = 200
ytick_int = 200
show_plt = True
save_imag = True
name_pred = model_weights_name.replace('weights', 'pred')
name_seg = model_weights_name.replace('weights', 'seg')
save_as = 'png'
post_process = True
post_process_only = False

# This file is similar to 'main_netw_predict_exec.py', it is mainly used
# to produce nicer plots for sets with more than 3 features (can be combined
# with the other file, it has not been done due to lack of time :-))


# Load data files
with np.load(file_pred + '.npz') as data:
    pred_X = data['data_X']
    if use_test_set == True:
        pred_Y = data['data_Y']
    else:
        y_true = None
    
_, height, width, channels = pred_X.shape

# Definition of various input parameters
model_args = (height, width, channels, n_classes)
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

# Perform prediction on new data
if post_process_only == False:
    if predict_all == True:
        prediction = my_model.model_predict(pred_X, verbose=1,
                                            return_format='list')
    else:
        prediction = my_model.model_predict(
                     pred_X[single_file_number, ...][np.newaxis, ...],
                     verbose=1,
                     return_format='list')
    # Prediction without mask plotting is for debugging purposes only
    if plot_prediction == True:
        for i in range(0, len(prediction)):
            if predict_all == True:
                file_num = i
            else:
                file_num = single_file_number
            helper_func.plot_predprob(prediction[i], n_classes=n_classes, 
                                      class_names=class_names,
                                      xtick_int=xtick_int, ytick_int=ytick_int,
                                      show_plt=show_plt, save_imag=save_imag,
                                      imag_name=name_pred + '_' + str(file_num),
                                      save_as=save_as)
    if plot_mask_with_input == True:
        for i in range(0, len(prediction)):
            if predict_all == True:
                file_num = i
                x_in = pred_X[i, ...]
            else:
                file_num = single_file_number
                x_in = pred_X[single_file_number, ...]
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask_3cl_input(prediction[i], x_in,
                                               y_true=y_true,
                                               class_to_plot=class_to_plot,
                                               input_to_plot=input_to_plot,
                                               input_name=input_name,
                                               xtick_int=xtick_int,
                                               ytick_int=ytick_int,
                                               show_plt=show_plt,
                                               save_imag=save_imag,
                                               imag_name=name_seg + '_inp_mask_' 
                                               + str(file_num),
                                               save_as=save_as)

if post_process == True:
    if predict_all == True:
        prediction_CRF = my_model.model_denseCRF_predict(pred_X, verbose=1,
                         return_format='list', **dense_CRF_kwargs)
    else:
        prediction_CRF = my_model.model_denseCRF_predict(
                         pred_X[single_file_number, ...][np.newaxis, ...],
                         verbose=1, return_format='list', **dense_CRF_kwargs)
    # Prediction without mask plotting is for debugging purposes only
    if plot_prediction == True:
        for i in range(0, len(prediction_CRF)):
            if predict_all == True:
                file_num = i
            else:
                file_num = single_file_number
            helper_func.plot_predprob(prediction_CRF[i], n_classes=n_classes,
                                      class_names=class_names,
                                      xtick_int=xtick_int, ytick_int=ytick_int,
                                      show_plt=show_plt,
                                      save_imag=save_imag,
                                      imag_name=name_pred + '_CRF_'
                                      + str(file_num),
                                      save_as=save_as)
    if plot_mask_with_input == True:
        for i in range(0, len(prediction_CRF)):
            if predict_all == True:
                file_num = i
                x_in = pred_X[i, ...]
            else:
                file_num = single_file_number
                x_in = pred_X[single_file_number, ...]
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask_3cl_input(prediction_CRF[i], x_in,
                                               y_true=y_true,
                                               class_to_plot=class_to_plot,
                                               input_to_plot=input_to_plot,
                                               input_name=input_name,
                                               xtick_int=xtick_int,
                                               ytick_int=ytick_int,
                                               show_plt=show_plt,
                                               save_imag=save_imag,
                                               imag_name=name_seg + '_inp_mask_CRF_' 
                                               + str(file_num),
                                               save_as=save_as)

del model, my_model, pred_X
    
    
  
