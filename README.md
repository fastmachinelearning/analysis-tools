# analysis-tools

To run the example below first copy the dataset from: https://cernbox.cern.ch/index.php/s/aGjXWDrDpugHeMf 

Convert input features and keras model predictions from a dataset to txt file

```
python convert_data_to_txt.py -c train_config_threelayer.yml -m model/KERAS_check_best_model.h5 -i data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z -t t_allpar_new -o data_to_txt
```

Run a test bench in Vivado that reads the input features and write on a file the outputs.
Find an example of test bench in the example-hls-test-bench folder.

Verify the accuracy of the fixed point NN output values against floating point expected values

```
python verify-hls-tb-outputs.py -i data_to_txt/KERAS_check_best_model_predictions.dat -I data_to_txt/KERAS_check_best_model_testbench_outputs.dat -t data_to_txt/KERAS_check_best_model_truth_labels.dat -o verify_hls
```

