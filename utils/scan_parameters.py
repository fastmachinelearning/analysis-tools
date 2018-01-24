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
import json
import yaml
import os
import argparse
import collections
import itertools
import subprocess
import xml.etree.ElementTree as ET
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ToRun = True

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

def ExtractFromXML(df, key, report_file):
    xmlpath = {
        "DSP48E" : ["AreaEstimates", "Resources", "DSP48E"],
        "Timing" : ["PerformanceEstimates", "SummaryOfTimingAnalysis", "EstimatedClockPeriod"],
        "Latency" : ["PerformanceEstimates", "SummaryOfOverallLatency", "Average-caseLatency"],
        "Interval" : ["PerformanceEstimates", "SummaryOfOverallLatency", "Interval-min"],
        "FF" : ["AreaEstimates", "Resources", "FF"],
        "LUT" : ["AreaEstimates", "Resources", "LUT"],
    }
    ## Add column to df
    for k in xmlpath:
        if k not in df:
            df[k] = None

    ## Check file exits
    if not os.path.isfile(report_file):
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

def ExtractROC(df, key, output_filename):
    truth_filename = "../data_to_txt_2NN/KERAS_check_best_model_truth_labels.dat"
    predict_filename = "../data_to_txt_2NN/KERAS_check_best_model_predictions.dat"

    if "AUC" not in df:
        df["AUC"] = None

    ## Check file exits
    if not os.path.isfile(output_filename):
        df.at[key, "AUC"] = None
        return False

    truth_df = pandas.read_csv(truth_filename, header=None)
    predict_df = pandas.read_csv(predict_filename, header=None)
    output_df = pandas.read_csv(output_filename, header=None)

    ## Expected AUC from keras
    # fpr, tpr, threshold = roc_curve(truth_df,predict_df)

    dfpr, dtpr, dthreshold = roc_curve(truth_df,output_df.iloc[:, 0])
    dauc = auc(dfpr, dtpr)
    df.at[key, "AUC"] = dauc

    plt.plot(dtpr,dfpr,label='%s point, auc = %.1f%%'%( key, dauc * 100))
    plt.savefig("%s.pdf" % key)


def RunProjs(projs):
    ##
    df = pandas.DataFrame.from_dict(projs, orient='index')
    # print(df.columns.values)
    for k, v in projs.items():
        PrepareYaml(k, v)
        pwd = os.getcwd()
        report_filename = "%s/../../keras-to-hls/%s/myproject_prj/solution1/syn/report/myproject_csynth.xml" % (pwd, k)
        output_filename = "%s/../../keras-to-hls/%s/myproject_prj/solution1/csim/build/res.dat" % (pwd,k)

        if ToRun and not os.path.exists(report_filename):
            ymltorun = "%s/%s.yml" % (pwd, k)
            outlog = open("%s.stdout" % k, 'w')
            errlog = open("%s.stderr" % k, 'w')
            subprocess.call("python keras-to-hls.py -c %s"  % ymltorun, cwd=r'../../keras-to-hls/',
                            stdout = outlog, stderr=errlog, shell=True)
            subprocess.call("ls", cwd=r'../../keras-to-hls/',
                            stdout = outlog, stderr=errlog, shell=True)
            subprocess.call("vivado_hls -f build_prj.tcl" , cwd=r'../../keras-to-hls/%s' % k,
                            stdout = outlog, stderr=errlog, shell=True)
            ## Remove large temp dir which consume a large disk space
            autopilot = "%s/../../keras-to-hls/%s/myproject_prj/solution1/.autopilot" % (pwd, k)
            if os.path.exists(autopilot):
                subprocess.call("rm -rf %s" % autopilot,
                            stdout = outlog, stderr=errlog, shell=True)

            ## Remove large data input file, whose size scale up with precision 
            wdbfile = "%s/../../keras-to-hls/%s/myproject_prj/solution1/sim/verilog/myproject.wdb" % (pwd, k)
            if os.path.exists(autopilot):
                subprocess.call("rm -rf %s" % autopilot,
                            stdout = outlog, stderr=errlog, shell=True)

        # Extract XML file
        ExtractFromXML(df, k,  report_filename)
        # Extract ROC curve
        ExtractROC(df, k, output_filename)

    return df

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        default = './scan.yml',
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: parser.error('A configuration file needs to be specified.')

    yamlConfig = parse_config(args.config)
    config = ReadConfig(yamlConfig)
    projs = FormVariation(config)
    df = RunProjs(projs)
    df.to_csv("output.csv")
