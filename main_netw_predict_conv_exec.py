
import netw_preprocess


main_path = r'~/...'
folder_predict = r'input'  # subfolder with images (or folders of images)
normalization = 'max'      # 'max' or 'mean'
filename = 'dataset_pred'  # name of computed output file (.npz)

netw_preprocess.predict_convert(main_path, folder_predict,
                                normalization=normalization,
                                filename=filename)
