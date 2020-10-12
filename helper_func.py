
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import resample

   
def plot_predprob(y_pred, class_names=None, n_classes=3, xtick_int=50,
                  ytick_int=50, show_plt=True, save_imag=True,
                  imag_name='pred_prob', save_as='pdf'):
    '''Function to plot predigted probability masks'''
    
    if class_names == None:
        class_names = [str(k) for k in range(1, y_pred.shape[-1]+1)]
    fig = plt.figure(figsize=(11, 4))
    grid = ImageGrid(fig, rect=[0.085, 0.07, 0.85, 0.9],
                     nrows_ncols=(1, n_classes),
                     axes_pad=0.25,
                     share_all=True,
                     cbar_location="right",
                     cbar_mode="single",
                     cbar_size="7%",
                     cbar_pad=0.2)

    for i in range(0, y_pred.shape[-1]):
        ax = grid[i]
        im = ax.imshow(y_pred[:, :, i], vmin=0, vmax=1, cmap='jet', 
                       interpolation='nearest')
        ax.set_xticks(np.arange(0, y_pred.shape[1]+1, xtick_int))
        ax.set_yticks(np.arange(0, y_pred.shape[0]+1, ytick_int))
        ax.set_xlabel(r'image width [pixel]')
        ax.set_ylabel(r'image height [pixel]')
        ax.set_title('Class ' + class_names[i])

    ax.cax.colorbar(im)
    ax.cax.toggle_label(True)
    if show_plt == True:
        plt.show()
    if save_imag == True:
        plt.savefig(imag_name + '.' + save_as, bbox_inches='tight')
        if show_plt == False:
            # Clear memory (or matplotlib history) although
            # the figure is not shown
            plt.close()
    

def plot_segmask(y_pred, y_true=None, class_to_plot=2, xtick_int=50,
                 ytick_int=50, show_plt=True, save_imag=True,
                 imag_name='pred_mask', save_as='pdf'):
    '''Function to plot segmentation mask (2 classes)'''
    
    m_temp = np.argmax(y_pred, axis=-1) + 1
    pred_mask = (m_temp*(m_temp==class_to_plot)) * (1.0/class_to_plot)
    # Plot prediction mask and ground truth
    if y_true is not None:
        g_temp = np.argmax(y_true, axis=-1) + 1
        gr_truth = (g_temp*(g_temp==class_to_plot)) * (1.0/class_to_plot)
        grid_cmap = ['jet', 'gray']
        grid_imag = [pred_mask, gr_truth]
        grid_title = [r'Segmentation mask', r'Ground truth']
        fig = plt.figure(figsize=(6.8, 4))
        grid = ImageGrid(fig, rect=[0.1, 0.07, 0.85, 0.9],
                     nrows_ncols=(1, 2),
                     axes_pad=0.25,
                     share_all=True)
        for i in range(0, 2):
            ax = grid[i]
            ax.imshow(grid_imag[i], vmin=0, vmax=1, cmap=grid_cmap[i])
            ax.set_xticks(np.arange(0, pred_mask.shape[1]+1, xtick_int))
            ax.set_yticks(np.arange(0, pred_mask.shape[0]+1, ytick_int))
            ax.set_xlabel(r'image width [pixel]')
            ax.set_ylabel(r'image height [pixel]')
            ax.set_title(grid_title[i])
    else:
        # Plot prediction mask
        plt.figure(figsize=(4.5, 4))
        plt.imshow(pred_mask, vmin=0, vmax=1, cmap='jet')
        plt.xticks(np.arange(0, pred_mask.shape[1]+1, xtick_int))
        plt.yticks(np.arange(0, pred_mask.shape[0]+1, ytick_int))
        plt.xlabel(r'image width [pixel]')
        plt.ylabel(r'image height [pixel]')
        plt.title(r'Segmentation mask')
        
    if show_plt == True:
        plt.show()
    if save_imag == True:
        plt.savefig(imag_name + '.' + save_as, bbox_inches='tight')
        if show_plt == False:
            # Clear memory (or matplotlib history) although the figure
            # is not shown
            plt.close()
    

def plot_segmask_input(y_pred, x_in, y_true=None, class_to_plot=2,
                        input_to_plot=1, input_name='channel 1',
                        xtick_int=50, ytick_int=50, show_plt=True,
                        save_imag=True, imag_name='pred_mask_input',
                        save_as='pdf'):
    '''Function to plot segmentation mask (2 classes)
       including 1 input channel'''
    
    m_temp = np.argmax(y_pred, axis=-1) + 1
    pred_mask = (m_temp*(m_temp==class_to_plot)) * (1.0/class_to_plot)
    if y_true is not None:
        # Plot prediction mask, ground truth and input image (1 channel)
        g_temp = np.argmax(y_true, axis=-1) + 1
        gr_truth = (g_temp*(g_temp==class_to_plot)) * (1.0/class_to_plot)
        grid_cmap = ['jet', 'gray', 'gray']
        grid_imag = [pred_mask, gr_truth, x_in[..., input_to_plot-1]]
        grid_title = [r'Segmentation mask', r'Ground truth', input_name]
        fig_width = 10.5
        n_cols = 3
    else:
        # Plot prediction mask and input image (1 channel)
        grid_cmap = ['jet', 'gray']
        grid_imag = [pred_mask, x_in[..., input_to_plot-1]]
        grid_title = [r'Segmentation mask', input_name]
        fig_width = 6.8
        n_cols = 2
        
    fig = plt.figure(figsize=(fig_width, 4))
    grid = ImageGrid(fig, rect=[0.1, 0.07, 0.85, 0.9],
                    nrows_ncols=(1, n_cols),
                    axes_pad=0.25,
                    share_all=True)
    for i in range(0, n_cols):
        ax = grid[i]
        ax.imshow(grid_imag[i], vmin=0, vmax=1, cmap=grid_cmap[i])
        ax.set_xticks(np.arange(0, pred_mask.shape[1]+1, xtick_int))
        ax.set_yticks(np.arange(0, pred_mask.shape[0]+1, ytick_int))
        ax.set_xlabel(r'image width [pixel]')
        ax.set_ylabel(r'image height [pixel]')
        ax.set_title(grid_title[i])
        
    if show_plt == True:
        plt.show()
    if save_imag == True:
        plt.savefig(imag_name + '.' + save_as, bbox_inches='tight')
        if show_plt == False:
            # Clear memory (or matplotlib history) although the figure
            # is not shown
            plt.close()


def plot_segmask_3cl_input(y_pred, x_in, y_true=None, class_to_plot=(2, 3),
                           input_to_plot=1, input_name='channel 1',
                           xtick_int=50, ytick_int=50, show_plt=True,
                           save_imag=True, imag_name='pred_mask_input',
                           save_as='pdf'):
    '''Function to plot segmentation mask (3 classes)
       including 1 input channel,
       this is a prototype version ==> can be combined with plot_segmask_input'''
    
    m_temp = np.argmax(y_pred, axis=-1) + 1
    pred_mask = ((m_temp*(m_temp==class_to_plot[0])) * (0.5/class_to_plot[0])
                ) + (
                 (m_temp*(m_temp==class_to_plot[1])) * (1.0/class_to_plot[1]))
    if y_true is not None:
        # Plot prediction mask, ground truth and input image (1 channel)
        g_temp = np.argmax(y_true, axis=-1) + 1
        gr_truth = ((g_temp*(g_temp==class_to_plot[0])) * (0.5/class_to_plot[0])
                   ) + (
                    (g_temp*(g_temp==class_to_plot[1])) * (1.0/class_to_plot[1]))
        grid_cmap = ['jet', 'gray', 'gray']
        grid_imag = [pred_mask, gr_truth, x_in[..., input_to_plot-1]]
        grid_title = [r'Segmentation mask', r'Ground truth', input_name]
        fig_width = 10.5
        n_cols = 3
    else:
        # Plot prediction mask and input image (1 channel)
        grid_cmap = ['jet', 'gray']
        grid_imag = [pred_mask, x_in[..., input_to_plot-1]]
        grid_title = [r'Segmentation mask', input_name]
        fig_width = 6.8
        n_cols = 2
        
    fig = plt.figure(figsize=(fig_width, 4))
    grid = ImageGrid(fig, rect=[0.1, 0.07, 0.85, 0.9],
                    nrows_ncols=(1, n_cols),
                    axes_pad=0.25,
                    share_all=True)
    for i in range(0, n_cols):
        ax = grid[i]
        ax.imshow(grid_imag[i], vmin=0, vmax=1, cmap=grid_cmap[i])
        ax.set_xticks(np.arange(0, pred_mask.shape[1]+1, xtick_int))
        ax.set_yticks(np.arange(0, pred_mask.shape[0]+1, ytick_int))
        ax.set_xlabel(r'image width [pixel]')
        ax.set_ylabel(r'image height [pixel]')
        ax.set_title(grid_title[i])
        
    if show_plt == True:
        plt.show()
    if save_imag == True:
        plt.savefig(imag_name + '.' + save_as, bbox_inches='tight')
        if show_plt == False:
            # Clear memory (or matplotlib history) although the figure
            # is not shown
            plt.close()
            

def bootstrap_mean_CI(data_samples, alpha=0.95, n_iterations=1000):
    '''Calculate the bootstrapped mean applying the percentile method to
       estimate a confidence interval (Efron method)'''
    
    stats = []
    for i in range(0, n_iterations):
        # Resample original samples
        data_resample = resample(data_samples, replace=True,
                                 n_samples=len(data_samples))
        # Calculate mean
        stats.append(np.mean(data_resample))
    # Confidence interval
    p_lower = ((1.0-alpha)/2.0) * 100
    p_upper = (alpha+((1.0-alpha)/2.0)) * 100
    lower_bound = max(0.0, np.percentile(stats, p_lower))
    upper_bound = min(1.0, np.percentile(stats, p_upper))
    # Print result
    print([lower_bound, upper_bound, (lower_bound+upper_bound)/2.0, 
          (upper_bound-lower_bound)/2])

    
    
if __name__ == "__main__":
    import sys
    plot_predprob(*sys.argv[1:])
    
    
