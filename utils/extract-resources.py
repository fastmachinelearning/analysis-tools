import numpy as np
import h5py
import os
import tarfile
import json
import argparse
import yaml
from shutil import copyfile

#######################################
## Config module
#######################################
def parse_config(config_file) :

    print "Loading configuration from " + str(config_file)
    config = open(config_file, 'r')
    return yaml.load(config)

############################################################################################
## M A I N
############################################################################################
def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("-c", action='store', dest='config',
                        help="Configuration file.")
    args = parser.parse_args()
    if not args.config: parser.error('A configuration file needs to be specified.')

    yamlConfig = parse_config(args.config)
    print "\n";
    print( 'project directory: {idir}'.format(idir=yamlConfig["ProjectDir"]) )
    print( 'top function: {tf}'.format(tf=yamlConfig["TopFunction"]) )

    report_filename = "%s/solution1/syn/report/%s_csynth.rpt" % ( yamlConfig["ProjectDir"],yamlConfig["TopFunction"] );
    report_file = open(report_filename,'r');

    # for line in report_file:
    #     print line

    lines = report_file.readlines();

    timingLines = [];
    latencyLines = [];
    resourceLines = [];
    for l in range(len(lines) - 2):
        # print l, lines[l]

        if "Timing" in lines[l] and "Summary" in lines[l+1]: 
            for il in range(l+2,l+7): timingLines.append( lines[il].strip() );

        if "Latency" in lines[l] and "Summary" in lines[l+1]: 
            for il in range(l+2,l+8): latencyLines.append( lines[il].strip() );

        if "Utilization Estimates" in lines[l] and "Summary" in lines[l+2]: 
            for il in range(l+3,l+6): resourceLines.append( lines[il].strip() );
            for il in range(l+14,l+16): resourceLines.append( lines[il].strip() );

    print "\n";
    for i in timingLines: print i;
    print "\n";
    for i in latencyLines: print i;
    print "\n";
    for i in resourceLines: print i;


if __name__ == "__main__":
    main();    
