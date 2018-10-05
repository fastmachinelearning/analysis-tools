import sys
import os
from optparse import OptionParser
from keras.models import load_model, Model
from keras.models import model_from_json
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
import math
import re

# To turn off GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''


def print_array_to_cpp(name, a, odir ):

    #count zeros
    zero_ctr = 0
    for x in np.nditer(a, order='C'):
        if x == 0: 
            zero_ctr += 1

    #put output in subdir for tarballing later
    f=open("prints/{}.h".format(name),"w")

    #meta data
    f.write("//Numpy array shape {}\n".format(a.shape))
    f.write("//Min {}\n".format(np.min(a)))
    f.write("//Max {}\n".format(np.max(a)))
    f.write("//Number of zeros {}\n".format(zero_ctr))
    f.write("\n")
    
    #c++ variable 
    if "w" in name: 
        f.write("weight_default_t {}".format(name))
    elif "b" in name: 
        f.write("bias_default_t {}".format(name))
    else:
        raise Exception('ERROR: Unkown weights type')

    #hls doesn't like 3d arrays... unrolling to 1d
    #also doing for all (including 2d) arrays now
    f.write("[{}]".format(np.prod(a.shape)))
    f.write(" = {")
    
    #fill c++ array.  
    #not including internal brackets for multidimensional case
    i=0
    for x in np.nditer(a, order='C'):
        if i==0:
            f.write("{}".format(x))
        else:
            f.write(", {}".format(x))
        i=i+1
    f.write("};\n")
    f.close()

    return zero_ctr


# The following two functions from
# https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print(g.file, '(File)', g.name)
 
    elif isinstance(g,h5py.Dataset) :
        print('(Dataset)', g.name, '    len =', g.shape) #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print('(Group)', g.name)
 
    else :
        print('WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name)
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).items() :
            subg = val
            print(offset, key)#, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')






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

    print_hdf5_file_structure(options.inputFile)

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
    jetImagePt = np.array(h5File.get('jetImage'))
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
        np.random.seed(12345)
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
    # jetImagePt = jetImagePt/maxVal[0]

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
    
    print('LOAD MODEL')
    #print(options.inputModel)
    #f = h5py.File(options.inputModel, 'r')
    #print("Keys: %s" % f.keys())
    #print_hdf5_file_structure(options.inputModel)
    #model = load_model(options.inputModel)
    #print('LOADED MODEL***********************************')
    
    #load model with json and h5 instead
    json_string = open('fromMaurizio/jetTagger_Conv2D_Small.json', 'r').read()
    model = model_from_json(json_string)
    model.load_weights('fromMaurizio/jetTagger_Conv2D_Small.h5')
    print('LOADED MODEL***********************************')

    #Export
    #model.save_weights('my_model_weights.h5')
    #outfile = open('model.json','w')
    #jsonString = model.to_json()
    #import json
    #with outfile:
    #    obj = json.loads(jsonString)
    #    json.dump(obj, outfile, sort_keys=True,indent=4, separators=(',', ': '))
    #    outfile.write('\n')

    predict_test = model.predict(x_image_test)

    print(x_image_test[0,:])
    print(x_image_test[0,:].shape)

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
    for e in range(x_image_test.shape[0]):
    #for e in range(0,0):
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


###############################
# inference
###############################

h5File = h5py.File('fromMaurizio/jetTagger_Conv2D_Small.h5')
print_hdf5_file_structure('fromMaurizio/jetTagger_Conv2D_Small.h5')

x_sample = x_image_test[0,:,:,:]
print(x_sample)
print_array_to_cpp("w_x_sample",x_sample,"xxx")

#batch norm
beta = h5File['/batch_normalization_1/batch_normalization_1/beta:0'][()]
gamma = h5File['/batch_normalization_1/batch_normalization_1/gamma:0'][()]
mean = h5File['/batch_normalization_1/batch_normalization_1/moving_mean:0'][()]
var = h5File['/batch_normalization_1/batch_normalization_1/moving_variance:0'][()]
scale = gamma/np.sqrt(var)
x_sample =(x_sample-mean)*scale+beta
print("OUT BATCH NORM")
print(x_sample)
print_array_to_cpp("w_x_sampleBN",x_sample,"xxx")


conv_k = h5File['/cnn2D_1_relu/cnn2D_1_relu/kernel:0'][()]
conv_b = h5File['/cnn2D_1_relu/cnn2D_1_relu/bias:0'][()]
conv2_k = h5File['/cnn2D_2_relu/cnn2D_2_relu/kernel:0'][()]
conv2_b = h5File['/cnn2D_2_relu/cnn2D_2_relu/bias:0'][()]
#dense_k = h5File['/dense_1_relu/dense_1_relu/kernel:0'][()]
#dense_b = h5File['/dense_1_relu/dense_1_relu/bias:0'][()]
#dense2_k = h5File['/output_softmax/output_softmax/kernel:0'][()]
#dense2_b = h5File['/output_softmax/output_softmax/bias:0'][()]


img_rows, img_cols = 25, 25
input_shape = (img_rows, img_cols, 1)

in_height = input_shape[0];
in_width  = input_shape[1];
in_chann  = input_shape[2];

f_height   = 3;
f_width    = 3;
f_outchann = 5; #number of filters

stride_width = 3;
stride_height = 3;
padding = "same"; #guess at the moment


# Derived
f_inchann  = in_chann;  #number of input channels
n_filters  = f_outchann;

# Padding
if (in_width % stride_width == 0):
    pad_along_width = max(f_width - stride_width, 0)
else:
    pad_along_width = max(f_width - (in_width % stride_width), 0)
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left


if (in_height % stride_height == 0):
    pad_along_height = max(f_height - stride_height, 0)
else:
    pad_along_height = max(f_height - (in_height % stride_height), 0)
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top

print("pad_left: {}".format(pad_left))
print("pad_right: {}".format(pad_right))
print("pad_top: {}".format(pad_top))
print("pad_bottom: {}".format(pad_bottom))

out_width  = math.ceil(float(in_width - f_width + 1) / float(stride_width))
out_height  = math.ceil(float(in_height - f_height + 1) / float(stride_height))
if padding == "same":
    out_width  = int(math.ceil(float(in_width) / float(stride_width)))
    out_height  = int(math.ceil(float(in_height) / float(stride_height)))
    in_width = in_width + pad_left + pad_right
    in_height = in_height + pad_top + pad_bottom

    print("in_width, post padding, should be: {}".format(in_width))
    print("in_height, post padding, should be: {}".format(in_height))

#    #f1 = open('pre.txt', 'w')
#    #np.savetxt(f1,x_sample)
    x_sample = np.pad(x_sample, [(pad_top,pad_bottom),(pad_left,pad_right),(0,0)], 'constant')
    print("x_sample shape: ",x_sample.shape)
#    #f2 = open('post.txt', 'w')
#    #np.savetxt(f2,x_sample)


conv_out = np.zeros((out_height,out_width,n_filters))

print("out_height: ",out_height)
print("out_width: ",out_width)

n_mult = 0
n_add = 0
for oh in range(0, out_height):
    for ow in range(0, out_width):
        for f in range(0, f_outchann): #n_filters                                                                                           
            channel_sum = 0;
            for c in range(0, in_chann):

                #count multiplications
                n_mult = n_mult + f_width*f_height

                #get filter
                my_filter = conv_k[:,:,c,f]

                #select data
                x_buffer = x_sample[:,:,c]
                x_buffer = x_buffer[oh*stride_height:oh*stride_height+f_height,ow*stride_width:ow*stride_width+f_width]

                #do multiplication
                my_mult = np.multiply(x_buffer, my_filter);

                #sum
                my_dot = np.sum(my_mult)
                channel_sum += my_dot

                if ow==0 and oh==0 and f==0 and c==0:
                    #if np.sum(x_buffer)>0 :
                    print("buffer shape: ",x_buffer.shape)
                    print("filter shape: ",my_filter.shape)
                    print("mult shape: ",my_mult.shape)
                    print("dot shape: ",my_dot.shape)
                    print("channel sum shape: ",channel_sum.shape)
                    print("buffer : ",x_buffer)
                    print("filter : ",my_filter)
                    print("mult : ",my_mult)
                    print("dot : ",my_dot)
                    print("channel sum : ",channel_sum)

            #print("conv_b[f] ",conv_b[f]
            conv_out[oh,ow,f] = channel_sum + conv_b[f]
            #print("conv_out[oh,ow,f] ",conv_out[oh,ow,f]  

print("n_mult: ",n_mult)
print("conv_out shape: ",conv_out.shape)
print("conv_out: ")
print(conv_out)

print_array_to_cpp("w_out",conv_out,"weights")

#Rest of network
conv_out = conv_out * (conv_out > 0) #relu                                                                                                  
print("RELU")
print(conv_out)

########################
## LAYER 2
########################


x_sample = conv_out



#batch norm 2
beta = h5File['/batch_normalization_2/batch_normalization_2/beta:0'][()]
gamma = h5File['/batch_normalization_2/batch_normalization_2/gamma:0'][()]
mean = h5File['/batch_normalization_2/batch_normalization_2/moving_mean:0'][()]
var = h5File['/batch_normalization_2/batch_normalization_2/moving_variance:0'][()]
scale = gamma/np.sqrt(var)
print("******* x_sample.shape ",x_sample.shape)
#x_sample[0,:] =(x_sample[0,:]-mean[0])*scale[0]+beta[0]
x_sample =(x_sample-mean)*scale+beta
print("OUT BATCH NORM")
print(x_sample)
print_array_to_cpp("w_x_sampleBN2",x_sample,"xxx")

in_height = conv_out.shape[0];
in_width  = conv_out.shape[1];
in_chann  = conv_out.shape[2];

print("x_sample.shape: ",x_sample.shape)
print("in_height: ",in_height)
print("in_width: ",in_width)
print("in_chann: ",in_chann)

f_height   = 5;
f_width    = 5;
f_outchann = 2; #number of filters

stride_width = 2;
stride_height = 2;
padding = "same";


# # Derived
f_inchann  = in_chann;  #number of input channels
n_filters  = f_outchann;

# # Padding
if (in_width % stride_width == 0):
    pad_along_width = max(f_width - stride_width, 0)
else:
    pad_along_width = max(f_width - (in_width % stride_width), 0)
pad_left = pad_along_width // 2
pad_right = pad_along_width - pad_left


if (in_height % stride_height == 0):
    pad_along_height = max(f_height - stride_height, 0)
else:
    pad_along_height = max(f_height - (in_height % stride_height), 0)
pad_top = pad_along_height // 2
pad_bottom = pad_along_height - pad_top

print("pad_left: {}".format(pad_left))
print("pad_right: {}".format(pad_right))
print("pad_top: {}".format(pad_top))
print("pad_bottom: {}".format(pad_bottom))

out_width  = math.ceil(float(in_width - f_width + 1) / float(stride_width))
out_height  = math.ceil(float(in_height - f_height + 1) / float(stride_height))
if padding == "same":
    out_width  = int(math.ceil(float(in_width) / float(stride_width)))
    out_height  = int(math.ceil(float(in_height) / float(stride_height)))
    in_width = in_width + pad_left + pad_right
    in_height = in_height + pad_top + pad_bottom
    
    print("in_width, post padding, should be: {}".format(in_width))
    print("in_height, post padding, should be: {}".format(in_height))
    
#     #f1 = open('pre.txt', 'w')
#     #np.savetxt(f1,x_sample)
    x_sample = np.pad(x_sample, [(pad_top,pad_bottom),(pad_left,pad_right),(0,0)], 'constant')
#     print("x_sample shape: ",x_sample.shape)
#     #f2 = open('post.txt', 'w')
#     #np.savetxt(f2,x_sample)


conv_out = np.zeros((out_height,out_width,n_filters))

print("out_height: ",out_height)
print("out_width: ",out_width)

n_mult = 0
n_add = 0
for oh in range(0, out_height):
    for ow in range(0, out_width):
        for f in range(0, f_outchann): #n_filters                                                                                           
            channel_sum = 0;
            for c in range(0, in_chann):

                #count multiplications
                n_mult = n_mult + f_width*f_height

                #get filter
                my_filter = conv2_k[:,:,c,f]
                
                #select data
                x_buffer = x_sample[:,:,c]
                x_buffer = x_buffer[oh*stride_height:oh*stride_height+f_height,ow*stride_width:ow*stride_width+f_width]

                #do multiplication
                my_mult = np.multiply(x_buffer, my_filter);

                #sum
                my_dot = np.sum(my_mult)
                channel_sum += my_dot

                if ow==0 and oh==0 and f==0 and c==0:
                    #if np.sum(x_buffer)>0 :
                    print("buffer shape: ",x_buffer.shape)
                    print("filter shape: ",my_filter.shape)
                    print("mult shape: ",my_mult.shape)
                    print("dot shape: ",my_dot.shape)
                    print("channel sum shape: ",channel_sum.shape)
                    print("buffer : ",x_buffer)
                    print("filter : ",my_filter)
                    print("mult : ",my_mult)
                    print("dot : ",my_dot)
                    print("channel sum : ",channel_sum)

            #print("conv_b[f] ",conv_b[f]
            conv_out[oh,ow,f] = channel_sum + conv2_b[f]
            #print("conv_out[oh,ow,f] ",conv_out[oh,ow,f]  

print("n_mult: ",n_mult)
print("conv_out shape: ",conv_out.shape)

print_array_to_cpp("w_out2",conv_out,"weights")

# #Rest of network
# conv_out = conv_out * (conv_out > 0) #relu                                                                                                  


# ###################

# conv_out = conv_out.flatten()
# dnn_out = np.dot(conv_out, dense_k)+dense_b
# dnn_out = dnn_out * (dnn_out > 0) #relu                                                                                                  


# dnn2_out = np.dot(dnn_out, dense2_k)+dense2_b
# dnn2_out = np.exp(dnn2_out) / sum(np.exp(dnn2_out)) #softmax

# print("Network output: ",dnn2_out)


######################3
##
##########################
#f3 = open('inference.txt', 'w')
#np.savetxt(f3, conv_out[:,:,:].flatten())

#conv_out = conv_out.flatten()
#print("flattened shape: ",conv_out.shape)

#dnn_out = np.dot(conv_out, dense_k)+dense_b
#
#f4 = open('dense.txt', 'w')
#np.savetxt(f4, dnn_out)
#print("Pre-softmax: ",dnn_out)
#
#dnn_out = np.exp(dnn_out) / sum(np.exp(dnn_out)) #softmax                                                                                 #  
#print("Network output: ",dnn_out)
#
#print_array_to_cpp("w_final_out",dnn_out,"weights")
#
#conv_only_keras()

