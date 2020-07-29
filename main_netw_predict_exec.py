
import json
import numpy as np
from keras.models import load_model
import helper_func
import netw_models
import netw_test_predict


file_pred = 'dataset_test'   # filename of input file (.npz)
n_classes = 3                # number of different labels / features (ground truth)
class_names = ["'boundary'", "'interior'", "'background'"]  # label names (only for headers in plot)
model_name = 'xxx'           # model name if a whole model is imported
model_weights_name = 'u_net_model_example_weights'  # name of weights set if only weights are imported (usual case)
load_entire_model = False    # if False then the provided set of weights is loaded, otherwise the whole model
predict_all = False          # True: parse all images in the .npz-file, False: only load image specified in the next line 
single_file_number = 17      # image to load when predict_all = False
plot_prediction = True       # plot probability map
plot_mask = True             # plot segmentation mask
include_input = True         # include input image in the mask plot
input_to_plot = 1            # which channel to include (e.g. phase image or ...)
input_name = r'Nuclear image'  # for header in plot
class_to_plot = 2            # which label / feature to plot along with the segmentation mask
use_test_set = True          # if True: the input array contains a ground truth, if False: without ground truth (for new images)
xtick_int = 50               # xtick interval in plot
ytick_int = 50               # ytick interval in plot
show_plt = True              # if False: plots are not shown
save_imag = False            # if False: plots are not saved
name_pred = model_weights_name.replace('weights', 'pred')  # name prefix of map plots
name_seg = model_weights_name.replace('weights', 'seg')    # name prefix of mask plots
save_as = 'png'              # save plots as
post_process = False         # perform dense crf post-processing
post_process_only = False    # return / compute only plots with dense crf post-processing

# This is a file for prototyping and not every possible combination of parameter
# settings is checked internally (some unusual combinations might return an error)


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
    if plot_mask == True and include_input == False:
        for i in range(0, len(prediction)):
            if predict_all == True:
                file_num = i
            else:
                file_num = single_file_number
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask(prediction[i], y_true=y_true,
                                     class_to_plot=class_to_plot,
                                     xtick_int=xtick_int,
                                     ytick_int=ytick_int,
                                     show_plt=show_plt,
                                     save_imag=save_imag,
                                     imag_name=name_seg + '_mask_' 
                                     + str(file_num),
                                     save_as=save_as)
    elif plot_mask == True and include_input == True:
        for i in range(0, len(prediction)):
            if predict_all == True:
                file_num = i
                x_in = pred_X[i, ...]
            else:
                file_num = single_file_number
                x_in = pred_X[single_file_number, ...]
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask_input(prediction[i], x_in,
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
    if plot_mask == True and include_input == False:
        for i in range(0, len(prediction_CRF)):
            if predict_all == True:
                file_num = i
            else:
                file_num = single_file_number
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask(prediction_CRF[i], y_true=y_true,
                                     class_to_plot=class_to_plot,
                                     xtick_int=xtick_int,
                                     ytick_int=ytick_int,
                                     show_plt=show_plt,
                                     save_imag=save_imag,
                                     imag_name=name_seg + '_mask_CRF_'
                                     + str(file_num),
                                     save_as=save_as)
    elif plot_mask == True and include_input == True:
        for i in range(0, len(prediction_CRF)):
            if predict_all == True:
                file_num = i
                x_in = pred_X[i, ...]
            else:
                file_num = single_file_number
                x_in = pred_X[single_file_number, ...]
            if use_test_set == True:
                y_true = pred_Y[file_num, ...]
            helper_func.plot_segmask_input(prediction_CRF[i], x_in,
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
    
    
  
