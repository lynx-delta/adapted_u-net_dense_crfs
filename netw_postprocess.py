
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def pp_denseCRF(imag, y_pred,
                pairwise_gauss=True, pairwise_bilateral=True,
                pw_gauss_sdims=(10, 10), pw_gauss_compat=3,
                pw_bilat_sdims=(20, 20), pw_bilat_schan=(0.005,),
                pw_bilat_compat=10, inf_steps=5):
    '''Dense CRF postprocessing using 2D image / softmax prediction matrix'''
    
    height, width, channels = imag.shape
    n_classes = y_pred.shape[-1]
    
    # Set unary energy
    d = dcrf.DenseCRF2D(width, height, n_classes)
    U = unary_from_softmax(np.moveaxis(y_pred, -1, 0))
    d.setUnaryEnergy(U)
    # Create the (color)-independent features and add them to the CRF
    if pairwise_gauss == True:    
        pw_gauss = create_pairwise_gaussian(sdims=pw_gauss_sdims,
                                            shape=y_pred.shape[:2])
        d.addPairwiseEnergy(pw_gauss, compat=pw_gauss_compat)
    # Create the (color)-dependent features and add them to the CRF
    if pairwise_bilateral == True:
        pw_bilateral = create_pairwise_bilateral(sdims=pw_bilat_sdims,
                                                 schan=pw_bilat_schan,
                                                 img=imag, chdim=2)
        d.addPairwiseEnergy(pw_bilateral, compat=pw_bilat_compat)
    # Inference
    Q = d.inference(inf_steps)
    # Reshape eigen matrix and return prediction in original shape 
    pred_Q = np.reshape(Q, (n_classes, height, width))
    pred_orig_shape = np.moveaxis(pred_Q, 0, -1)
    return pred_orig_shape



if __name__ == "__main__":
    import sys
    pp_denseCRF(*sys.argv[1:])





