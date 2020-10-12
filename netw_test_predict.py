
import numpy as np
from sklearn.metrics import f1_score
import netw_postprocess


class TrainedModel():
    '''Class for testing and prediction'''
    
    def __init__(self, model):
        '''Initialization'''
        # Prototype (further development ==> inherit class directly from
        # keras.models.Model)
        self.model = model
           
    def model_test(self, test_X, test_Y, verbose=1, average=None):
        '''
        Evaluate model on a test dataset
        test_X: input data (numpy array)
        test_Y: input labels (numpy array, one-hot encoded)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        average: None = F1 score per class, 'micro' = globally averaged
                 default = None
        returns: [metric_overall_acc, metric_f1_score]
        '''
        y_pred = self.model.predict(test_X, batch_size=1, verbose=verbose)
        # Return metrics
        return self._calc_metrics(test_Y, y_pred, average)
                  
    def model_predict(self, pred_X, verbose=1, return_format='list'):
        '''
        Use model for prediction (unseen data)
        pred_X: input data (numpy array)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        return_format: 'list' = return list, 'array' = return numpy array
                       default = 'list'
        returns: prediction (probabilities for each class)
                 as list or numpy array (specified by return_format)
        '''
        y_pred = self.model.predict(pred_X, batch_size=1, verbose=verbose)
        if return_format == 'list': # mainly for debugging purpose
            y_pred_list = []
            for i in range(0, y_pred.shape[0]):
                y_pred_list.append(y_pred[i, ...])
            return y_pred_list
        else:
            return y_pred
             
    def model_denseCRF_test(self, test_X, test_Y, verbose=1, average=None,
                            pairwise_gauss=True, pairwise_bilateral=True,
                            pw_gauss_sdims=(10, 10), pw_gauss_compat=3,
                            pw_bilat_sdims=(20, 20), pw_bilat_schan=(0.005,),
                            pw_bilat_compat=10, inf_steps=5):
        '''
        Evaluate model with CRF post-processing on a test dataset
        test_X: input data (numpy array)
        test_Y: input labels (numpy array, one-hot encoded)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        average: None = F1 score per class, 'micro' = globally averaged
                 default = None
        pairwise_gauss: True = add pairwise Gaussian energy to the CRF
                        False = CRF without pairwise Gaussian energy
                        default = True
        pairwise_bilateral: True = add pairwise bilateral energy to the CRF
                            False = CRF without pairwise bilateral energy
                            default = True
        pw_gauss_sdims: "strength" of the location content
                        default = (10, 10)
        pw_gauss_compat: "strength" of potential
                         default = 3
        pw_bilat_sdims: "strength" of the location content
                        default = (20, 20)
        pw_bilat_schan: "strength" of the image content
                        default = (0.005,)
        pw_bilat_compat: "strength" of potential
                         default = 10
        inf_steps: inference, number of steps
                   default = 5
        returns: [metric_overall_acc, metric_f1_score]
        '''
        y_pred = self.model.predict(test_X, batch_size=1, verbose=verbose)
        # Perform CRF post-processing (infer posterior probabilities)
        y_pred_crf = np.zeros(y_pred.shape, dtype='float32')
        for i in range(0, y_pred.shape[0]):
            y_pred_crf[i, ...] = netw_postprocess.pp_denseCRF(
                                 test_X[i, ...], y_pred[i, ...],
                                 pairwise_gauss, pairwise_bilateral,
                                 pw_gauss_sdims, pw_gauss_compat,
                                 pw_bilat_sdims, pw_bilat_schan,
                                 pw_bilat_compat, inf_steps)
        # Return metrics
        return self._calc_metrics(test_Y, y_pred_crf, average)
         
    def model_denseCRF_predict(self, pred_X, verbose=1, return_format='list',
                               pairwise_gauss=True, pairwise_bilateral=True,
                               pw_gauss_sdims=(10, 10), pw_gauss_compat=3,
                               pw_bilat_sdims=(20, 20), pw_bilat_schan=(0.005,),
                               pw_bilat_compat=10, inf_steps=5):
        '''
        Use model with CRF post-processing for prediction (unseen data)
        pred_X: input data (numpy array)
        verbose: 0 = no output, 1 = progress bar
                 default = 1
        return_format: 'list' = return list, 'array' = return numpy array
                       default = 'list'
        pairwise_gauss: True = add pairwise Gaussian energy to the CRF
                        False = CRF without pairwise Gaussian energy
                        default = True
        pairwise_bilateral: True = add pairwise bilateral energy to the CRF
                            False = CRF without pairwise bilateral energy
                            default = True
        pw_gauss_sdims: "strength" of the location content
                        default = (10, 10)
        pw_gauss_compat: "strength" of potential
                         default = 3
        pw_bilat_sdims: "strength" of the location content
                        default = (20, 20)
        pw_bilat_schan: "strength" of the image content
                        default = (0.005,)
        pw_bilat_compat: "strength" of potential
                         default = 10
        inf_steps: inference, number of steps
                   default = 5
        returns: prediction (probabilities for each class)
                 as list or numpy array (specified by return_format)
        '''
        y_pred = self.model.predict(pred_X, batch_size=1, verbose=verbose)
        # Perform CRF post-processing (infer posterior probabilities)
        if return_format == 'list': # mainly for debugging purpose
            y_pred_crf_list = []
            for i in range(0, y_pred.shape[0]):
                y_pred_crf_list.append(netw_postprocess.pp_denseCRF(
                                       pred_X[i, ...], y_pred[i, ...],
                                       pairwise_gauss, pairwise_bilateral,
                                       pw_gauss_sdims, pw_gauss_compat,
                                       pw_bilat_sdims, pw_bilat_schan,
                                       pw_bilat_compat, inf_steps))
            return y_pred_crf_list
        else:
            y_pred_crf = np.zeros(y_pred.shape, dtype='float32')
            for i in range(0, y_pred.shape[0]):
                y_pred_crf[i, ...] = netw_postprocess.pp_denseCRF(
                                     pred_X[i, ...], y_pred[i, ...],
                                     pairwise_gauss, pairwise_bilateral,
                                     pw_gauss_sdims, pw_gauss_compat,
                                     pw_bilat_sdims, pw_bilat_schan,
                                     pw_bilat_compat, inf_steps)
            return y_pred_crf
 
    def _calc_metrics(self, y_true, y_pred, average):
        '''Calculate metrics (overall accuracy, F1 score)'''
        y_pred_max = np.argmax(y_pred, axis=-1)
        y_true_max = np.argmax(y_true, axis=-1)
        # Overall (pixelwise) accuracy
        # (for multiclass classification the same as Jaccard index)
        metric_overall_acc = np.sum(np.equal(
                                    y_true_max,  y_pred_max)) / y_true_max.size
        # F1 score
        metric_f1_score = f1_score(y_true_max.flatten(), y_pred_max.flatten(),
                                   average=average)
        return [metric_overall_acc, metric_f1_score]

