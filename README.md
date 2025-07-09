# cell_segment_quality_cnn
Simple CNN for classifying segmentation quality of individual cells (as part of a wider 3D sample) 

WIP!

- [Notebook](extract_cell_data.ipynb) to extract ROIs and plots of ROIs with segmentation overlays
- [segmentation quality GUI](rate_cell_gui.py) for generating classification labels (ratings.json) 
- [Training script](train_example.py) to train model with Datasets containing ratings and 2-channel ROIs (mask and fluorescence channel)
- [Inference script](eval_script.py) to get CNN classifier output (i.e., segmentation ratings between 0 and 1) 
- [CNN Model Code](model.py)
