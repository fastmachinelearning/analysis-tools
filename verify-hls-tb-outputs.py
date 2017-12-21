import ROOT
from ROOT import *
from argparse import ArgumentParser
from optparse import OptionParser
import sys, time, os
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetErrorX(0.5)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option('-i','--inputFloat'   ,action='store',type='string',dest='inputFloat'   ,default='data_to_txt/KERAS_check_best_model_predictions.dat', help='input file with floating point values')
    parser.add_option('-I','--inputTb'   ,action='store',type='string',dest='inputTb'   ,default='data_to_txt/KERAS_check_best_model_testbench_outputs.dat', help='input file with HLS test bench values')
    parser.add_option('-t','--inputTruth'   ,action='store',type='string',dest='inputTruth'   ,default='data_to_txt/KERAS_check_best_model_truth_labels.dat', help='input file with truth labels for ROC curves')
    parser.add_option('-o','--output'   ,action='store',type='string',dest='outputDir'   ,default='verify_hls/', help='output directory')
    (options,args) = parser.parse_args()

    if os.path.isdir(options.outputDir):
        print "Directory exist: do not create"
    else:
        os.mkdir(options.outputDir)
            
    finTB = open(options.inputTb,'r')
    finTB_lines = finTB.readlines()
    print "Found",len(finTB_lines),"events in file",options.inputTb

    finFloat = open(options.inputFloat,'r')
    finFloat_lines = finFloat.readlines()
    print "Found",len(finFloat_lines),"events in file",options.inputFloat

    finTruth = open(options.inputTruth,'r')
    finTruth_lines = finTruth.readlines()
    print "Found",len(finTruth_lines),"events in file",options.inputTruth

    hout = ROOT.TH1F("hout","",400,-20,20)
    hout2d = ROOT.TH2F("hout2d","",100,0,1,400,-20,20)
    labels_truth = np.zeros(shape=(len(finFloat_lines),5))
    prediction_fixed = np.zeros(shape=(len(finFloat_lines),5))
    prediction_float = np.zeros(shape=(len(finFloat_lines),5))

    for e in range(len(finTB_lines)):

     out_fl_array = finFloat_lines[e].replace(' \n','').split(' ')
     out_tb_array = finTB_lines[e].replace(' \n','').split(' ')
 
     lab_truth_array = finTruth_lines[e].replace(' \n','').split(' ')
     for l in range(len(lab_truth_array)): labels_truth[e][l] = float(lab_truth_array[l])
  
     for o in range(len(out_tb_array)):
 
      out_fl = float(out_fl_array[o])
      out_tb = float(out_tb_array[o])
  
      hout2d.Fill(out_fl,out_fl-out_tb)
      if out_fl != 0:
       hout.Fill((out_fl-out_tb)/out_fl)
       hout2d.Fill(out_fl,(out_fl-out_tb)/out_fl)   
      elif out_fl == 0 and out_tb != 0:
       hout.Fill((out_fl-out_tb)/out_tb)
       hout2d.Fill(out_fl,(out_fl-out_tb)/out_tb)
      elif out_fl == 0 and out_tb == 0:
       hout.Fill(0)
       hout2d.Fill(out_fl,0)
     
       prediction_fixed[e][o] = out_tb
       prediction_float[e][o] = out_fl


    c_2 = ROOT.TCanvas("c_2","c_2")
    c_2.cd()
    c_2.SetLogy()
    hout.SetLineColor(kBlack)
    hout.SetLineWidth(2)
    hout.GetXaxis().SetTitle("(Float_point - Fixed_point)/Float_point")
    hout.Draw("HIST")
    c_2.SaveAs('verify_hls/diff_numbers_rel.png')

    c_3 = ROOT.TCanvas("c_3","c_3")
    c_3.cd()
    c_3.SetLogz()
    hout2d.GetXaxis().SetTitle("Prediction float point value")
    hout2d.GetYaxis().SetTitle("(Float_point - Fixed_point)/Float_point")
    hout2d.Draw("COLZ")
    c_3.SaveAs('verify_hls/diff_numbers_2d.png')

    fpr = {}
    tpr = {}
    auc1 = {}
    labels = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
    colors = ['b', 'g', 'r', 'c', 'm']

    for l in range(len(labels)):

     plt.figure(3+l)

     label = labels[l]

     plt.title('%s tagger'%label)

     fpr[label], tpr[label], threshold = roc_curve(labels_truth[:,l],prediction_fixed[:,l])
     auc1[label] = auc(fpr[label], tpr[label])
     plt.plot(tpr[label],fpr[label],label='fixed point',color=colors[l],linestyle='--')

     fpr[label], tpr[label], threshold = roc_curve(labels_truth[:,l],prediction_float[:,l])
     auc1[label] = auc(fpr[label], tpr[label])
     plt.plot(tpr[label],fpr[label],label='floating point',color=colors[l])
 
     plt.semilogy()
     plt.xlabel("sig. efficiency")
     plt.ylabel("bkg. mistag rate")
     plt.ylim(0.001,1)
     plt.grid(True)
     plt.legend(loc='upper left')
     plt.savefig('verify_hls/%s_ROC.png'%(label))
