import sys
import os
from optparse import OptionParser
from keras.models import load_model, Model
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
import h5py
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
import yaml
# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

## Config module
def parse_config(config_file) :

    #print "Loading configuration from " + str(config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-m','--model'   ,action='store',type='string',dest='inputModel'   ,default='model/KERAS_check_best_model.h5', help='input model')
    parser.add_option('-i','--input'   ,action='store',type='string',dest='inputFile'   ,default='data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z', help='input file')
    parser.add_option('-t','--tree'   ,action='store',type='string',dest='tree'   ,default='t_allpar_new', help='tree name')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='data_to_txt/', help='output directory')
    parser.add_option('-c','--config'   ,action='store',type='string', dest='config', default='train_config_threelayer.yml', help='configuration file')
    (options,args) = parser.parse_args()
     
    #yamlConfig = parse_config(options.config)

    if os.path.isdir(options.outputDir):
        print("Directory exist: do not create")
    else:
        os.mkdir(options.outputDir)


    # To use one data file:
    h5File = h5py.File(options.inputFile)
    treeArray = h5File[options.tree][()]

    # List of features to use
    # List of features to use
    #features = yamlConfig['Inputs']
    
    # List of labels to use
    #labels = yamlConfig['Labels']

    # Convert to dataframe
    #features_df = pd.DataFrame(treeArray,columns=features)
    #labels_df = pd.DataFrame(treeArray,columns=labels)

    #Copy processing from Maurizio    
    jets = h5File.get('jets')
    jetImagePt = np.array(h5File.get('jetImagePt'))
    print(jets.shape, jetImagePt.shape)

    # prepare target
    target = jets[:,-6:-1]
    print(target.shape)
    
    # prepare expert-feature array
    expFeature = jets[:,1:4]
    print(expFeature.shape)

    def shuffle(a, b, c):
        iX = a.shape[1]
        iY = a.shape[2]
        b_shape = b.shape[1]
        a = a.reshape(a.shape[0], iX*iY)
        total = np.column_stack((a,b))
        total = np.column_stack((total,c))
        np.random.shuffle(total)
        a = total[:,:iX*iY]
        b = total[:,iX*iY:iX*iY+b_shape]
        c = total[:,iX*iY+b_shape:]
        a = a.reshape(a.shape[0], iX, iY, 1)
        return a,b,c

    # shuffle datasets
    jetImagePt, expFeature, target = shuffle(jetImagePt, expFeature, target)

    from sklearn.preprocessing import MinMaxScaler
    ## normalize expert features inputs and outputs between 0 and 1
    scaler = MinMaxScaler()
    print(scaler.fit(expFeature))
    maxVal = scaler.data_max_
    minVal = scaler.data_min_
    expFeature = (expFeature-minVal)/(maxVal-minVal)
    # normalize the images dividing by the maximum pT
    jetImagePt = jetImagePt/maxVal[0]

    iSplit = int(0.7*target.shape[0])
    x_image_train = jetImagePt[:iSplit, :, :]
    x_image_test = jetImagePt[iSplit:, :, :]
    x_train = expFeature[:iSplit, :]
    x_test = expFeature[iSplit:, :]
    y_train = target[:iSplit, :]
    y_test = target[iSplit:, :]
    
    print(x_image_train.shape)

    print('END PROCESSING')

    # Convert to numpy array with correct shape
    #features_val = features_df.values
    #labels_val = labels_df.values

    #X_train_val, X_test, y_train_val, y_test = train_test_split(features_val, labels_val, test_size=0.2, random_state=42)
    #print X_train_val.shape
    #print y_train_val.shape
    #print X_test.shape
    #print y_test.shape

    #Normalize
    #if yamlConfig['NormalizeInputs']:
    # scaler = preprocessing.StandardScaler().fit(X_train_val)
    # X_test = scaler.transform(X_test) 
             
    modelName = options.inputModel.split('/')[-1].replace('.h5','')
    
    #print options.inputModel
    model = load_model(options.inputModel)
    print('LOADED MODEL***********************************')
    predict_test = model.predict(x_image_test)

    print("Writing",y_test.shape[1],"predicted labels for",y_test.shape[0],"events in outfile",(options.outputDir+'/'+modelName+'_truth_labels.dat'))  
    outf_labels = open(options.outputDir+'/'+modelName+'_truth_labels.dat','w')
    for e in range(y_test.shape[0]):
     line=''
     for l in range(y_test.shape[1]):
      line+=(str(y_test[e][l])+' ')
     outf_labels.write(line+'\n')
    outf_labels.close() 
        
    print("x_image_test shape",x_image_test.shape)
    print("Writing",x_image_test.shape[1],"x",x_image_test.shape[2],"x",x_image_test.shape[3],"features for",x_image_test.shape[0],"events in outfile",(options.outputDir+'/'+modelName+'_input_features.dat'))
    outf_features = open(options.outputDir+'/'+modelName+'_input_features.dat','w')
    #for e in range(x_image_test.shape[0]):
    for e in range(0,100):
        line=''
        for r in range(0,x_image_test.shape[1]):
            for c in range(0,x_image_test.shape[2]):
                for ch in range(0,x_image_test.shape[3]):
                    line+=(str(x_image_test[e][r][c][ch])+' ')
        outf_features.write(line+'\n')    
    outf_features.close()  
     
    print("Writing",predict_test.shape[1],"predicted labels for",predict_test.shape[0],"events in outfile",(options.outputDir+'/'+modelName+'_predictions.dat'))
    outf_predict = open(options.outputDir+'/'+modelName+'_predictions.dat','w')
    for e in range(predict_test.shape[0]):
     line=''
     for l in range(predict_test.shape[1]):
      line+=(str(predict_test[e][l])+' ')
     outf_predict.write(line+'\n')
    outf_predict.close() 
