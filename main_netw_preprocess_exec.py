
import netw_preprocess


main_path = r'~/...'
folder_train = r'data_training'    # folder names with datasets
folder_val = r'data_validation'
folder_test = None #r'data_test'   # if None: folder is ignored
n_classes = 3                      # number of labels / features in ground truth
select_patches = True              # performs random patch selection and stores the patches in a newly created folder
change_dir = True                  # if True: get the data from newly created folder (usual case if patch selection is performed)
n_patches_train = 30               # patches per image (training set)
n_patches_val = 30                 # patches per image (validation set)
n_patches_test = 30                # patches per image (test set)
augment_data = True                # perform data augmentation (is only applied to training set)
augment_steps = 1                  # default = 1
patch_train_kwargs = {'patch_shape': (256, 256),      # arguments training set
                      'new_folder_ext': 'patches',
                      'change_dir': change_dir}
patch_val_kwargs = {'patch_shape': (256, 256),        # arguments validation set
                    'new_folder_ext': 'patches',
                    'change_dir': change_dir}
patch_test_kwargs = {'patch_shape': (256, 256),       # arguments test set
                    'new_folder_ext': 'patches',
                    'change_dir': change_dir}
aug_kwargs = {'rotate_prop': 0.2,                     # settings data augmentation
              'hflip_prop': 0.2,
              'vflip_prop': 0.2,
              'scale_prop': 0.0, 'scale_factor': 1.2,
              'shear_prop': 0.0, 'shear_factor': 0.3}
train_kwargs = {'normalization': 'max',               # image normalization
                'filename': 'dataset_train'}          # name of .npz-file
val_kwargs = {'normalization': 'max',
              'filename': 'dataset_val'}
test_kwargs = {'normalization': 'max',
               'filename': 'dataset_test'}

# Prepare training data
if folder_train is not None:
    prep_train = netw_preprocess.DataPreprocessor(main_path, folder_train)
    # Use patches for training
    if select_patches == True:
        prep_train.select_patches(n_patches_train, **patch_train_kwargs)
    # Augmentation
    if augment_data == True:
        for i in range(0, augment_steps):
            prep_train.augment_data(**aug_kwargs)
    # Save to file
    prep_train.data_to_array(n_classes, **train_kwargs)
    
# Prepare validation data
if folder_val is not None:
    prep_val = netw_preprocess.DataPreprocessor(main_path, folder_val)
    # Use patches for validation
    if select_patches == True:
        prep_val.select_patches(n_patches_val, **patch_val_kwargs)
    # Save to file
    prep_val.data_to_array(n_classes, **val_kwargs)
    
# Prepare test data
if folder_test is not None:
    prep_test = netw_preprocess.DataPreprocessor(main_path, folder_test)
    # Use patches for testing
    if select_patches == True:
        prep_test.select_patches(n_patches_test, **patch_test_kwargs)
    # Save to file
    prep_test.data_to_array(n_classes, **test_kwargs)





