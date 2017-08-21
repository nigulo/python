# Plotting the Mount Wilson HK public data
# from *.mgd files
# Plotting syntax: python plotmw.py <mgd file path>
# Jyri Lehtinen - 20.9.2016

import sys
import numpy as np
import matplotlib.pyplot as pl

#-------------------------------------------------------------------------#

def read_data():
    '''Read in the JD and Mound Wilson S-index values,
    transform the JD into non-truncated form by adding back
    the truncation of -2444000 d,
    return observing time in both JD and year'''
    try:
        data=sys.argv[1]                         # datafile
        g=file(data,'r')
        rows=g.readlines()
        g.close()
    except:
        print '  Give a *.mgd data file!'
        sys.exit()

    mwos=[]                                      # Mount Wilson S-index
    jd=[]                                        # Julian Date
    try:
        star=rows[0].split()[0]                  # Star ID
        for row in rows:
            row=row.split()
            mwos.append(float(row[1]))
            jd.append(float(row[2]))
    except:
        print '  Wrong format file! Should be *.mgd'
        sys.exit()
    mwos=np.array(mwos)
    jd=np.array(jd) + 2444000
    year=jd/365.25-4712

    return jd,year,mwos,star

def plot_mwos(year,mwos,star):
    ax=pl.axes([0.12,0.12,0.8,0.8])
    pl.plot(year,mwos,'k.')

    pl.xlim([year[0]-0.5,year[-1]+0.5])
    mins=min(mwos)
    maxs=max(mwos)
    ds=maxs-mins
    pl.ylim([mins-0.1*ds,maxs+0.1*ds])
    pl.xlabel('year')
    pl.ylabel('S')
    pl.title(star)
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    pl.show()

#-------------------------------------------------------------------------#

def main():
    jd,year,mwos,star=read_data()
    plot_mwos(year,mwos,star)

main()
