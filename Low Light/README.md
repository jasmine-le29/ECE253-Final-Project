First run resnet_training. It was created in Jupyter Notebook, and the file is annotated with the cells I used. It will download the dataset ExDark and train it automatically.
Next is resnet_test. The line "test_dir = ..." must be changed to whatever the folder of testing images is. The accuracy will be calculated.
The computation of LPIPS, NIQE, and BRISQUE were done using the LPDM github: https://github.com/savvaki/LPDM
