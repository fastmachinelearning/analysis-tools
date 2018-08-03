import pandas
import matplotlib
import matplotlib.pyplot as plt

df = pandas.read_csv("output_BenConv2d.csv")
#print(df)

df.loc[:,'Precision'] = df['DefaultPrecision'].str.strip("ap_fixed")
print(df.loc[:,'Precision'])

plotlist = ['AUC0','AUC1','AUC2','AUC3','AUC4']

fig, ax = plt.subplots()
ax.autoscale(True, axis='y')
for t in plotlist:
    ax.clear()
    maxx = minn = []
    for i in range(1, 2):
        dff = df[df.ReuseFactor == i]
        dff.loc[:,'sort'] = dff.Precision.str.extract('<(\s*\d+),', expand =False).astype(int)
        dff.sort_values('sort', inplace=True, ascending=True)
        print(dff[t])
        maxx.append(dff[t].max())
        minn.append(dff[t].min())
        dff.plot('Precision', t, marker='o', ax = ax, title =t, label= "Reuse%d" % i)
    #ax.set_ylim( 0.5*min(minn), 1.5*max(maxx))
    pltt = ax.get_figure()
    pltt.savefig("%s.pdf" % t)
    pltt.savefig("%s.png" % t)

