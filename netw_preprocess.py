
from os import listdir, makedirs
from os.path import isfile, isdir, join
import random
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from skimage import io
from skimage import transform as trf


class DataPreprocessor():
    '''Class for data preprocessing'''
    
    def __init__(self, main_path, folder_name):
        '''Initialization'''
        self.main_path = main_path
        self.folder_name = folder_name
        self.path = join(self.main_path, self.folder_name)
        subdir = [f for f in listdir(self.path) if isdir(join(self.path, f))]
        if any(x in subdir[0] for x in ['data', 'input', 'Data', 'Input']):
            data_folder_path = join(self.path, subdir[0])
            label_folder_path = join(self.path, subdir[1])
        else:
            data_folder_path = join(self.path, subdir[1])
            label_folder_path = join(self.path, subdir[0])
        # Get path of input data folders
        subdir = [f for f in listdir(data_folder_path) if isdir(join(
                data_folder_path, f))]
        self.input_path = []
        if len(subdir) == 0:
            self.input_path.append([data_folder_path])
        else:
            for i in range(0, len(subdir)):
                temp_path = join(data_folder_path, subdir[i])
                self.input_path.append([temp_path])
        # Get path of label data folders
        subdir = [f for f in listdir(label_folder_path) if isdir(join(
                label_folder_path, f))]
        self.label_path = []
        if len(subdir) == 0:
            self.label_path.append([label_folder_path])
        else:
            for i in range(0, len(subdir)):
                temp_path = join(label_folder_path, subdir[i])
                self.label_path.append([temp_path])
    
    
    def select_patches(self, n_patches, patch_shape=(256, 256), 
                       new_folder_ext='patches', change_dir=False):
        '''Select patches in all images (provided in folders)'''
        self._init_folders()
        # Initialize new directories (for patches)
        patch_input_path = self._new_dir(self.input_path, self.folder_name, 
                                         self.folder_name + '_' + new_folder_ext)
        patch_label_path = self._new_dir(self.label_path, self.folder_name, 
                                         self.folder_name + '_' + new_folder_ext)
        # Patch selection
        prog = -1
        for file, counter in self._get_file_generator_patches(patch_input_path,
                                                              patch_label_path):
            # Generate random number pairs
            if prog != counter:
                image = io.imread('{}\{}'.format(file[0], file[1]))
                row, col = random.randint(0, (
                                          image.shape[0]-patch_shape[0])-1
                                          ), random.randint(0, (
                                          image.shape[1]-patch_shape[1])-1)
                pair_seen = set((row, col))
                rand_pair = [[row, col]]
                for n in range(1, n_patches):
                    row, col = random.randint(0, (
                                              image.shape[0]-patch_shape[0])-1
                                              ), random.randint(0, (
                                              image.shape[1]-patch_shape[1])-1)
                    while (row, col) in pair_seen:
                        row, col = random.randint(0, (
                                                  image.shape[0]-patch_shape[0])-1
                                                  ), random.randint(0, (
                                                  image.shape[1]-patch_shape[1])-1)
                    pair_seen.add((row, col))
                    rand_pair.append([row, col])
                prog = counter
            # Perform random patch selection
            self._random_patch_selection(file, rand_pair, patch_shape)
        
        if change_dir == True:
            self.input_path = patch_input_path
            self.label_path = patch_label_path
            self.path = self.path.replace(self.folder_name,
                                          self.folder_name + '_' + new_folder_ext)
            
        
    def augment_data(self, rotate_prop=0.2, hflip_prop=0.2, vflip_prop=0.2,
                     scale_prop=0.2, scale_factor=1.2, shear_prop=0.2,
                     shear_factor=0.3):
        '''Perform data augmentation in all subfolders'''
        self._init_folders()
        data_len = len(self.input_data[0])
        # Rotation
        if rotate_prop is not None and rotate_prop > 0.0:
            n_files = int(data_len*rotate_prop)
            file_list = [random.randint(0, data_len-1) for n in range(0, n_files)]
            # Perform rotation
            for file, counter in self._get_file_generator(file_list):
                image = io.imread('{}\{}'.format(file[0], file[1]))
                rotate_tf = trf.SimilarityTransform(rotation=np.deg2rad(180))
                new_imag = trf.warp(image, inverse_map=rotate_tf, mode='reflect', 
                           preserve_range=True).astype(image.dtype)
        
                io.imsave(fname='{}\{}'.format(
                        file[0], 'rot_' + str(
                                counter) + '_' + file[1]), arr=new_imag)
        # Horizontal flip
        if hflip_prop is not None and hflip_prop > 0.0:
            n_files = int(data_len*hflip_prop)
            file_list = [random.randint(0, data_len-1) for n in range(0, n_files)]
            # Perform rotation
            for file, counter in self._get_file_generator(file_list):
                image = io.imread('{}\{}'.format(file[0], file[1]))
                new_imag = image[:, ::-1]
        
                io.imsave(fname='{}\{}'.format(
                        file[0], 'hfl_' + str(
                                counter) + '_' + file[1]), arr=new_imag)
        # Vertical flip
        if vflip_prop is not None and vflip_prop > 0.0:
            n_files = int(data_len*vflip_prop)
            file_list = [random.randint(0, data_len-1) for n in range(0, n_files)]
            # Perform rotation
            for file, counter in self._get_file_generator(file_list):
                image = io.imread('{}\{}'.format(file[0], file[1]))
                new_imag = image[::-1, :]
        
                io.imsave(fname='{}\{}'.format(
                        file[0], 'vfl_' + str(
                                counter) + '_' + file[1]), arr=new_imag)
        # Zooming
        if scale_factor is None or scale_factor < 1:
            scale_factor = 1.0
        if scale_prop is not None and scale_prop > 0.0:
            n_files = int(data_len*scale_prop)
            file_list = [random.randint(0, data_len-1) for n in range(0, n_files)]
            # Perform rotation
            for file, counter in self._get_file_generator(file_list):
                image = io.imread('{}\{}'.format(file[0], file[1]))
                new_imag = trf.rescale(
                        image, scale_factor, mode='reflect', preserve_range=True)
                left = int((new_imag.shape[0]-image.shape[0])/2)
                right = left+image.shape[0]
                bottom = int((new_imag.shape[1]-image.shape[0])/2)
                top = bottom+image.shape[1]
                crop_imag = new_imag[bottom:top, left:right].astype(image.dtype)
                
                io.imsave(fname='{}\{}'.format(
                        file[0], 'scl_' + str(
                                counter) + '_' + file[1]), arr=crop_imag)
        # Shearing
        if shear_factor is None or shear_factor < 0.1 or shear_factor > 0.5:
            shear_factor = 0.3
        if shear_prop is not None and shear_prop > 0.0:
            n_files = int(data_len*shear_prop)
            file_list = [random.randint(0, data_len-1) for n in range(0, n_files)]
            # Perform rotation
            for file, counter in self._get_file_generator(file_list):
                image = io.imread('{}\{}'.format(file[0], file[1]))
                affine_tf = trf.AffineTransform(shear=shear_factor)
                new_imag = trf.warp(image, inverse_map=affine_tf, mode='reflect', 
                           preserve_range=True).astype(image.dtype)
                
                io.imsave(fname='{}\{}'.format(
                        file[0], 'shr_' + str(
                                counter) + '_' + file[1]), arr=new_imag)
        
        
    def data_to_array(self, n_classes, normalization='max', filename='dataset'):
        '''Fetch data from folder and save data (X) array
           and label (Y) array in one common .npz-file'''
        self._init_folders()
        # Load first data set (to determine input shape)
        single_input = self._get_input_data(0, normalization)
        # Initialize array for input data
        data_X = np.zeros(
                (len(self.input_data[0]), *single_input.shape), dtype='float32')
        # Fill input data array
        for i in range(data_X.shape[0]):
            data_X[i, :, :, :] = self._get_input_data(i, normalization)    
        # Determine label input type
        if len(self.label_path) == 1:
            one_hot_encoder = self._get_label_data_single
        else:
            one_hot_encoder = self._get_label_data_mult
        # Initialize array for input data
        data_Y = np.zeros(
                (len(self.label_data[0]), single_input.shape[0],
                                          single_input.shape[1],
                                          n_classes),
                                          dtype='float32')
        # Fill label data array
        for i in range(data_Y.shape[0]):
            data_Y[i, :, :, :] = one_hot_encoder(i, n_classes)
        # Save datasets, labelsets to file
        np.savez(join(self.path, filename + '.npz'), data_X=data_X, data_Y=data_Y)
        del data_X, data_Y  # free up memory
    
    
    def _new_dir(self, old_path, old_folder_name, new_folder_name):
        '''Create new directories'''
        new_path = []
        for item in old_path:
            temp_str = item[0]
            temp_dir = temp_str.replace(old_folder_name, new_folder_name)
            makedirs(temp_dir)
            new_path.append([temp_dir])
        return new_path
    
    
    def _init_folders(self):
        '''Read files in folders'''
        # Get files in input data folders
        self.input_data = []
        for i in range(0, len(self.input_path)):
            self.input_data.append([f for f in listdir(
                    self.input_path[i][0]) if isfile(
                            join(self.input_path[i][0], f))])
        # Get files in label data folders
        self.label_data = []
        for i in range(0, len(self.label_path)):
            self.label_data.append([f for f in listdir(
                    self.label_path[i][0]) if isfile(
                            join(self.label_path[i][0], f))])        
        # Check if all folders contain the same number of files
        it = iter(self.input_data + self.label_data)
        len_entry = len(next(it))
        if not all(len(n) == len_entry for n in it):
            raise ValueError('Not all lists have same length!')
        
    
    def _get_file_generator_patches(self, patch_input_path, patch_label_path):
        '''Generator to retrieve file names for patch selection'''
        n_folders = len(self.input_path + self.label_path)
        n_files = len(self.input_data[0])
        temp_path = n_files * (self.input_path + self.label_path)
        temp_patch_path = n_files * (patch_input_path + patch_label_path)
        temp_files = [(self.input_data + self.label_data)[j][i]
                      for i in range(0, n_files)
                      for j in range(0, n_folders)]
        for i in range(0, len(temp_files)):
            yield (temp_path[i][0], temp_files[i], 
                   temp_patch_path[i][0]), divmod(i, n_folders)[0]
            
            
    def _get_file_generator(self, file_list):
        '''Generator to retrieve file names'''
        n_folders = len(self.input_path + self.label_path)
        temp_path = len(file_list) * (self.input_path + self.label_path)
        temp_files = [(self.input_data + self.label_data)[j][i]
                      for i in file_list
                      for j in range(0, n_folders)]
        for i in range(0, len(temp_files)):
            yield (temp_path[i][0], temp_files[i]), divmod(i, n_folders)[0]
            
    
    def _random_patch_selection(self, file, rand_pair, patch_shape):
        '''Selects patches (randomly) from original image'''
        image = io.imread('{}\{}'.format(file[0], file[1]))
        for n in range(0, len(rand_pair)):
            # Crop patch n
            patch_n = image[rand_pair[n][0]:(rand_pair[n][0] + patch_shape[0]), 
                            rand_pair[n][1]:(rand_pair[n][1] + patch_shape[1])
                            ].astype(image.dtype)
            # Save patch n    
            io.imsave(fname='{}\{}'.format(
                      file[2], 'patch_' + str(n) + '_' + file[1]), arr=patch_n)

        
    def _get_input_data(self, k, normalization):
        '''Read data files and return separate numpy arrays'''
        input_arrays = []
        for h in range(0, len(self.input_data)):
            image = io.imread('{}\{}'.format(
                    self.input_path[h][0], self.input_data[h][k])
                             )
            # Normalize image
            if normalization == 'max':
                image = image / image.max()
            elif normalization == 'mean':
                image = image / image.mean()
            if len(image.shape) != 3:
                image = image[:, :, np.newaxis]
            input_arrays.append(image)
        single_input = np.concatenate(input_arrays, axis=2)
        return single_input
    
    
    def _get_label_data_single(self, k, n_classes):
        '''Read label files and perform one-hot encoding,
        single file which contains all classes''' 
        image = io.imread('{}\{}'.format(
                    self.label_path[0][0], self.label_data[0][k]))
        if len(image.shape) != 3:
            image = image[:, :, np.newaxis]
        if image.max() >= 255:   # if image is black / white (8, 16 or 24 bit)
            image = image / image.max()
        image = np.around(image)  # round float type data (data augmentation)
        one_hot_labels = []
        for g in range(0, n_classes):
            class_i = np.ones(image.shape, dtype='float32') * (image == g)
            one_hot_labels.append(class_i)
        # Move background class to last index
        one_hot_labels = one_hot_labels[1:] + [one_hot_labels[0]]
        return np.concatenate(one_hot_labels, axis=2)
        
        
    def _get_label_data_mult(self, k, _):
        '''Read label files and perform one-hot encoding,
        multiple files with 2 classes each (one common class (background) is
        present in every image)'''
        label_arrays = []
        for g in range(0, len(self.label_data)):
            image = io.imread('{}\{}'.format(
                    self.label_path[g][0], self.label_data[g][k]))
            if len(image.shape) != 3:
                image = image[:, :, np.newaxis]
            if image.max() >= 255:   # if image is black / white (8, 16 or 24 bit)
                image = image / image.max()
            image = np.around(image)  # round float type data (data augmentation)
            label_arrays.append(image)
        # Load first label image
        imag_ref = label_arrays[0]
        # Perform one-hot encoding
        one_hot_labels = [imag_ref]
        for g in range(1, len(self.label_data)):
            imag_mult = imag_ref * label_arrays[g]
            imag_corr = label_arrays[g] - imag_mult
            one_hot_labels.append(imag_corr)
            imag_ref = imag_ref + imag_corr
        one_hot_labels.append(np.ones(imag_ref.shape)-imag_ref)
        return np.concatenate(one_hot_labels, axis=2)

    
def predict_convert(main_path, folder_name, normalization='max',
                    filename='dataset'):
    '''
    Data coverter (to .npz file) for images to predict
    main_path: path to main folder
    folder_name: data folder with images (can contain subfolders)
    normalization: divides image by image max or mean
                    default = 'max'
    filename: filename of created .npz file
              default = 'dataset'
    '''
    # Initialze directory
    path = join(main_path, folder_name)
    subdir = [f for f in listdir(path) if isdir(join(path, f))]
    input_path = []
    if len(subdir) == 0:
        input_path.append([path])
    else:
        for i in range(0, len(subdir)):
            temp_path = join(path, subdir[i])
            input_path.append([temp_path])
    # Initialize folders
    input_data = []
    for i in range(0, len(input_path)):
        input_data.append([f for f in listdir(input_path[i][0]) if isfile(
                           join(input_path[i][0], f))])
    # Check if all folders contain the same number of files
    it = iter(input_data)
    len_entry = len(next(it))
    if not all(len(n) == len_entry for n in it):
        raise ValueError('Not all lists have same length!')
    # Load first data set (to determine input shape)
    single_input = get_data(0, input_path, input_data, normalization)
    data_X = np.zeros((len(input_data[0]), *single_input.shape), 
                      dtype='float32')
    # Fill input data array
    for i in range(data_X.shape[0]):
        data_X[i, :, :, :] = get_data(i, input_path, input_data, normalization)
    # Save dataset to file
    np.savez(join(path, filename + '.npz'), data_X=data_X)
    del data_X  # free up memory

    
def get_data(k, input_path, input_data, normalization):
    '''Read data files and return numpy array'''
    input_arrays = []
    for h in range(0, len(input_data)):
        image = io.imread('{}\{}'.format(
                input_path[h][0], input_data[h][k]))
        # Normalize image
        if normalization == 'max':
            image = image / image.max()
        elif normalization == 'mean':
            image = image / image.mean()
        if len(image.shape) != 3:
            image = image[:, :, np.newaxis]
        input_arrays.append(image)
    single_input = np.concatenate(input_arrays, axis=2)
    return single_input   
    

    
if __name__ == "__main__":
    import sys
    predict_convert(*sys.argv[1:])    
