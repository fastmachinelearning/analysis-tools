#!/usr/bin/env python
# encoding: utf-8

# File        : scan_parameters.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2017 Dec 26
#

from __future__ import print_function
from sklearn.metrics import roc_curve, auc
import pprint
import pandas
import numpy as np
import json
import yaml
import os, sys
import argparse
import collections
import itertools
import subprocess
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ToRun = True
CleanUp = False
xmlpath = {
    "DSP48E" : ["AreaEstimates", "Resources", "DSP48E"],
    "Timing" : ["PerformanceEstimates", "SummaryOfTimingAnalysis", "EstimatedClockPeriod"],
    "Latency" : ["PerformanceEstimates", "SummaryOfOverallLatency", "Average-caseLatency"],
    "Interval" : ["PerformanceEstimates", "SummaryOfOverallLatency", "Interval-min"],
    "FF" : ["AreaEstimates", "Resources", "FF"],
    "LUT" : ["AreaEstimates", "Resources", "LUT"],
}
prediction = None

#######################################
## Config module
#######################################
def parse_config(config_file) :
    print ("Loading configuration from " + str(config_file))
    config = open(config_file, 'r')
    return yaml.load(config)

def ReadConfig(yamlConfig):
    dicts  = {}
    scandict = []
    for k in yamlConfig.keys():
        dicts[k] = None
        if isinstance(yamlConfig[k], list):
            dicts[k] = {i :i for i in yamlConfig[k]}
        else:
            dicts[k] = yamlConfig[k]
    pprint.pprint(dicts)
    return dicts


def FormVariation(Config):
    ## Damn, can't think of a smart way
    ## Getting the variables for variation
    var = {}
    for k, v in Config.items():
        ## Use ScanXXX as the config variation for the scan. 
        ## Skip in the config
        if "Scan" == k[:4]:
            continue
        if isinstance(v, dict):
            var[k] = v

    ## Form different projs
    projs = {}
    for i in itertools.product(*var.values()):
        # print(i)
        proj = {}
        namesup = ""
        for k, v in Config.items():
            if k not in var:
                proj[k] = v
            else:
                idx = var.keys().index(k)
                proj[k] = v[list(i)[idx]]
                namesup +="_%s%s" % (k, list(i)[idx])
        proj["OutputDir"] = proj["OutputDir"] + namesup
        # pprint.pprint(proj)
        projs[proj["OutputDir"]] = proj
    return projs


def PrepareYaml(name, proj):
    with open("%s.yml" % name, 'w') as file:
        yaml.dump(proj, file, default_flow_style=False)

def ConfigureTestBench(inputdata):
    fout = open('tmp.cpp','w')
    for line in open('../example-hls-test-bench/myproject_test.cpp','r').readlines():
        fout.write(line.replace('tb_input_data.dat', inputdata))
    fout.close()
    os.system('mv tmp.cpp myproject_test.cpp')

def PassCSim(key):

    stdout = key+'.stdout'
    if not os.path.isfile(stdout): return False

    if 'CSim failed' in open(stdout,'r').read():
     print("CSim failed with errors for ",key)
     return False

    if '*** C/RTL co-simulation finished: FAIL ***' in open(stdout,'r').read():
     print("C/RTL co-simulation failed for ",key)
     return False

    return True

def ExtractFromXML(df, key, report_file):
    global xmlpath
    ## Add column to df
    for k in xmlpath:
        if k not in df:
            df[k] = None

    ## Check file exits
    if not os.path.isfile(report_file):
        print("file: ",report_file,"does not exist! Synthesis failed!")
        for k, v in xmlpath.items():
            df.at[key, k] = None
        return False

    ## Parsing XML file
    tree = ET.parse(report_file)
    root = tree.getroot()
    for k, v in xmlpath.items():
        path = root
        for i in v:
            path = path.find(i)
        df.at[key, k] = path.text

def ExtractROC(df, key, output_filename, config):
    global prediction
    truth_filename = os.path.abspath(config["ScanTruth"])
    predict_filename = os.path.abspath(config["ScanPrediction"])

    truth_df = np.loadtxt(truth_filename)
    predict_df = np.loadtxt(predict_filename)
    output_df = np.loadtxt(output_filename)
    Noutputs = 1
    if truth_df.ndim > 1:
      Noutputs = truth_df.shape[1]

    if prediction is None:
        temp_truth = pandas.DataFrame(truth_df,
            columns = ["truth_%d" % x for x in range(Noutputs)])
        temp_predict = pandas.DataFrame(predict_df,
            columns = ["predict_%d" % x for x in range(Noutputs)])
        prediction = temp_truth.join(temp_predict)

    temp_output = pandas.DataFrame(output_df,
        columns = ["%s_%d" % (key, x) for x in range(Noutputs)])
    prediction = prediction.join(temp_output)

    for i in range(Noutputs):
        if "AUC%i"%(i) not in df:
            df["AUC%i"%i] = None
        if "ExpAUC%i"%i not in df:
            df["ExpAUC%i"%i] = None
        ## Check file exits
        if not os.path.isfile(output_filename):
            df.at[key, "AUC%i"%i] = None
            df.at[key, "ExpAUC%i"%i] = None
            continue
         ## Check csim passed w/o errors
        if not PassCSim(key):
            df.at[key, "AUC%i"%i] = None
            df.at[key, "ExpAUC%i"%i] = None
            continue

    for i in range(Noutputs):
        ## Expected AUC from keras
        if Noutputs > 1:
          efpr, etpr, ethreshold = roc_curve(truth_df[:,i],predict_df[:,i])
        else:
          efpr, etpr, ethreshold = roc_curve(truth_df[:],predict_df[:])
        eauc = auc(efpr, etpr)
        df.at[key, "ExpAUC%i"%i] = eauc
        ## Expected AUC from HLS
        if Noutputs>1:
          dfpr, dtpr, dthreshold = roc_curve(truth_df[:,i],output_df[:,i])
        else:
          dfpr, dtpr, dthreshold = roc_curve(truth_df[:],output_df[:])
        dauc = auc(dfpr, dtpr)
        df.at[key, "AUC%i"%i] = dauc



def RunProjs(projs, config):
    hlsdir = config["ScanHLS4MLDir"]
    ##
    df = pandas.DataFrame.from_dict(projs, orient='index')
    for k, v in projs.items():
        PrepareYaml(k, v)
        pwd = os.getcwd()
        report_filename = "%s/keras-to-hls/%s/%s_prj/solution1/syn/report/myproject_csynth.xml" %\
                (hlsdir, k, config["ProjectName"])
        output_filename = "%s/keras-to-hls/%s/%s_prj/solution1/csim/build/tb_output_data.dat" % \
                (hlsdir,k, config["ProjectName"])

        if ToRun and not os.path.exists(report_filename):
            print("Running %s" % k)
            ymltorun = "%s.yml" % k
            outlog = open("%s.stdout" % k, 'w')
            errlog = open("%s.stderr" % k, 'w')

            ConfigureTestBench(os.path.abspath(config["ScanInputData"]))

            subprocess.call("cp %s.yml %s/keras-to-hls/" % (k, hlsdir),
                            stdout = outlog, stderr=errlog, shell=True)
            subprocess.call("python keras-to-hls.py -c %s"  % ymltorun, cwd=r'%s/keras-to-hls/'%hlsdir,
                            stdout = outlog, stderr=errlog, shell=True)
            subprocess.call('cp myproject_test.cpp build_prj.tcl %s/keras-to-hls/%s'%(hlsdir,k),
                            stdout = outlog, stderr=errlog, shell=True)
            subprocess.call("vivado_hls -f build_prj.tcl" , cwd=r'%s/keras-to-hls/%s' %(hlsdir,k),
                            stdout = outlog, stderr=errlog, shell=True)

#============================================================================#
#-----------------------------     Clean Up     -----------------------------#
#============================================================================#
            if CleanUp:
                ## Remove large temp dir which consume a large disk space
                autopilot = "%s/keras-to-hls/%s/myproject_prj/solution1/.autopilot" % (hlsdir,k)
                if os.path.exists(autopilot):
                    subprocess.call("rm -rf %s" % autopilot,
                                stdout = outlog, stderr=errlog, shell=True)

                subprocess.call('mkdir %s/keras-to-hls/results_%s'%(hlsdir,k),
                                stdout = outlog, stderr=errlog, shell=True)

                if os.path.exists(output_filename):
                    subprocess.call('cp %s %s/keras-to-hls/results_%s/.'%(output_filename,hlsdir,k),
                                    stdout = outlog, stderr=errlog, shell=True)
                    output_filename = '%s/keras-to-hls/results_%s/tb_output_data.dat'%(hlsdir,k)

                if os.path.exists(report_filename):
                    subprocess.call('cp %s %s/keras-to-hls/results_%s/.'%(report_filename,hlsdir,k),
                                    stdout = outlog, stderr=errlog, shell=True)
                    report_filename = '%s/keras-to-hls/results_%s/myproject_csynth.xml'%(hlsdir,k)


                subprocess.call('rm -rf %s/keras-to-hls/%s'%(hlsdir,k),
                                stdout = outlog, stderr=errlog, shell=True)


        # Extract XML file
        ExtractFromXML(df, k,  report_filename)
        # Extract ROC curve
        ExtractROC(df, k, output_filename,config)

    return df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        default = './scan.yml',
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: 
      parser.error('A configuration file needs to be specified.')

    yamlConfig = parse_config(args.config)
    config = ReadConfig(yamlConfig)
    projs = FormVariation(config)
    df = RunProjs(projs, config)
    df.to_csv("output_%s.csv" % config["OutputDir"])
    prediction.to_pickle("predict_%s.pkl" % config["OutputDir"])
