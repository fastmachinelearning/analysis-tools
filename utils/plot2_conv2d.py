#!/usr/bin/env python
# encoding: utf-8

# File        : PlotParameterScan.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2018 Mar 07
#
# Description : 

# In[1]:

import pandas
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("classic")

dfmap = {
    #"2016.4"         : pandas.read_csv("output_Can2016p2.csv" ),
    # "2017.2 Before" : pandas.read_csv("output_Can2017p2.csv" ),
    # "2017.2 After"  : pandas.read_csv("output_1LayerNorm.csv" ),
    #"2017.2 High"    : pandas.read_csv("output_Can2017p2High.csv" ),
    #"2017.4"         : pandas.read_csv("output_Can2017p4.csv" ),
    # "1Layer"          : pandas.read_csv("HLS4ML/output_1LayerNorm.csv"),
    # "3LayerFull"     : pandas.read_csv("HLS4ML/output_3LayerFull.csv"),
    # "3LayerPrune"    : pandas.read_csv("HLS4ML/output_3LayerPrunI6.csv"),
    #"3LayerPruneSAT4"    : pandas.read_csv("output_3LayerPrunSAT4.csv"),
    #"3LayerPruneNoSAT4"    : pandas.read_csv("output_3LayerPrunNoSAT4.csv"),
    # "3LayerPruneNoSAT5"    : pandas.read_csv("output_3LayerPrunNoSAT5.csv"),
    # "3LayerPruneSAT6"    : pandas.read_csv("output_3LayerPrunSAT6.csv"),
    # "3LayerPrune"    : pandas.read_csv("HLS4ML/output_3LayerPrun.csv"),
    # "3LayerPruneI6"    : pandas.read_csv("HLS4ML/output_3LayerPrunI6.csv"),
    # "3LayerPruneI4"    : pandas.read_csv("HLS4ML/output_3LayerPrunI6.csv"),
    "BenConv2d"        : pandas.read_csv("output_BenConv2d.csv"),
}

Tagorder = ['j_g', 'j_q', 'j_w', 'j_z', 'j_t']
# plotlist = ['FracAUC0', 'FracAUC1', 'FracAUC2', 'FracAUC3', 'FracAUC4']
#plotlist = ['DSP*Reuse', 'G-ops', 'DSP48E', 'Latency', 'Efficiency', 'LUT', 'Interval', 'FF', 'Timing']

def ProcessDF(df):
    df.loc[:,'Precision'] = df['DefaultPrecision'].str.strip("ap_fixed")
    df.loc[:, 'G-ops'] = df['DSP48E'] * 0.2/df['Interval']
    df.loc[:, 'DSP*Reuse'] = df['DSP48E'] * df.ReuseFactor
    df.loc[:,'sort'] = df.Precision.str.extract('<(\s*\d+),', expand =False).astype(int)
    ## Calculate Efficiency
    xilinxDSPMap ={
        "xc7vx690tffg1927-2" : 3600,
        "xcku115-flvf1924-2-i" : 5520,
    }
    df['MaxDSP48E'] = df.XilinxPart
    df=df.replace({'MaxDSP48E': xilinxDSPMap})
    # print df.MaxDSP48E
    df.loc[:, 'Efficiency'] = df['G-ops']/ (df.MaxDSP48E*0.2)

    for x in df.columns:
        if "ExpAUC" in x:
            fracname = x.replace("Exp", "Frac")
            df.loc[:, fracname] = df[x.strip("Exp")]/df[x]
            print(fracname)
    return df


def PlotAUC(title,df):
    fig, ax = plt.subplots()
    ax.autoscale(True, axis='y')
    for i in [1]:
    # for i in df.ReuseFactor.unique():
        ax.clear()
        legs =[]
        maxx = minn = []
        for t in [x for x in df.columns if "FracAUC" in x]:
            dff = df[(df.ReuseFactor == i) & (df.XilinxPart == 'xcku115-flvf1924-2-i')]
            dff.sort_values('sort', inplace=True, ascending=True)
            maxx.append(dff[t].max())
            minn.append(dff[t].min())
            label = Tagorder[int(t[-1])]
            dff.plot('Precision', t, marker='o', ax = ax, label= label)
            legs.append(label)
        ax.set_ylim( 0.5*min(minn), 1.2*max(maxx))
        ax.grid(True)
        ax.set_title("%s Reuse Factor=%s" % (title, i), loc='right')
        if t == "DSP48E":
            plt.axhline(5520, color='r')
            plt.text(15, 5521, "Max DSP48E", color='r')
        # ax.set_title("Vivado_HLS v%s" % version, loc='right')
        ax.set_title("HLS4ML Preliminary", loc='left', fontname ="Arial", size=15)
        ax.set_ylabel("Fraction AUC", size=15, horizontalalignment='right')
        ax.set_xlabel('Precision', size=15, horizontalalignment='right')
        ax.legend(legs, loc="best", borderpad=1.1, labelspacing=1.1, prop={'size':12})
        # ax.set_yscale('log')
        pltt = ax.get_figure()
        #pltt.legend()
        pltt.savefig("AUC_Reuse%s_%s.png" % (i, title.replace('.','p')))

def PlotDFvsResue(title,df):
    fig, ax = plt.subplots()
    ax.autoscale(True, axis='y')
    for t in plotlist:
        print(t)
        ax.clear()
        legs =[]
        maxx = minn = []
        for i in df.ReuseFactor.unique():
            dff = df[(df.ReuseFactor == i) & (df.XilinxPart == 'xcku115-flvf1924-2-i')]
            dff.sort_values('sort', inplace=True, ascending=True)
            maxx.append(dff[t].max())
            minn.append(dff[t].min())
            label ="Reuse Factor=%d" % i
            # label ="Reuse%d_%s" % (i, t)
            dff.plot('Precision', t, marker='o', ax = ax, label= label)
            legs.append(label)
        ax.set_ylim( 0.5*min(minn), 1.3*max(maxx))
        ax.grid(True)
        ax.set_title("%s Kintex" % title, loc='right')
        if t == "DSP48E":
            plt.axhline(5520, color='r')
            plt.text(15, 5521, "Max DSP48E", color='r')
        # ax.set_title("Vivado_HLS v%s" % version, loc='right')
        ax.set_title("HLS4ML Preliminary", loc='left', fontname ="Arial", size=15)
        ax.set_ylabel(t, size=15, horizontalalignment='right')
        ax.set_xlabel('Precision', size=15, horizontalalignment='right')
        ax.legend(legs, loc="best", borderpad=1.1, labelspacing=1.1, prop={'size':12})
        # ax.set_yscale('log')
        pltt = ax.get_figure()
        #pltt.legend()
        pltt.savefig("%s_v%s.png" % (t, title.replace('.','p')))


if __name__ == "__main__":
    for k, df in dfmap.items():
        df = ProcessDF(df)
        # PlotDFvsResue(k,df)
        PlotAUC(k, df)


# In[10]:

def CompDF(title, dfs):
    plotlist = ['DSP*Reuse', 'G-ops',  'DSP48E', 'Latency', 'LUT', 'Interval', 'FF', 'Timing']
    #plotlist = ['DSP*Reuse', 'G-ops', 'AUC', 'DSP48E', 'Latency', 'Efficiency', 'LUT', 'Interval', 'FF', 'Timing']
    fig, ax = plt.subplots()
    ax.autoscale(True, axis='y')
    for t in plotlist:
        ax.clear()
        legs=[]
        maxx = minn = []
        for label in sorted(dfs.keys()):
            dff = dfs[label]
        #for label, dff in dfs.items():
            dff.loc[:,'sort'] = dff.Precision.str.extract('<(\s*\d+),', expand =False).astype(int)
            dff.sort_values('sort', inplace=True, ascending=True)
            maxx.append(dff[t].max())
            minn.append(dff[t].min())
            dff.plot('Precision', t, marker='o', ax = ax, label= label)
            legs.append(label)
        ax.set_ylim( 0.5*min(minn), 1.3*max(maxx))
        pltt = ax.get_figure()
        ax.grid(True)
        ax.set_title(title, loc='right')
        ax.set_title("HLS4ML Preliminary", loc='left', fontname ="Arial", size=15)
        ax.set_ylabel(t, size=15, horizontalalignment='right')
        ax.set_xlabel('Precision', size=15, horizontalalignment='right')
        ax.legend(legs, loc="best", borderpad=1.1, labelspacing=1.1, prop={'size':12})
        #ax.legend(legs, loc="upper left", borderpad=1.1, labelspacing=1.1, prop={'size':12})
        pltt = ax.get_figure()
        pltt.savefig("Comp%s_%s.pdf" % (title, t))
        pltt.savefig("Comp%s_%s.png" % (title, t))
        
dfs = {}
for k,df in dfmap.items():
    dfs[ k] = df[(df.ReuseFactor == 1) & (df.XilinxPart == 'xcku115-flvf1924-2-i') ]
    #dfs["v%s Reuse3" % k] = df[(df.ReuseFactor == 3) & (df.XilinxPart == 'xcku115-flvf1924-2-i') ]
    #dfs["v%s Reuse6" % k] = df[(df.ReuseFactor == 6) & (df.XilinxPart == 'xcku115-flvf1924-2-i') ]
#CompDF("3Layer",dfs)
#CompDF("Kintex",dfs)
#CompDF("Vertex7",dfs)
