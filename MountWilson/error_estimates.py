import numpy as np
import os
import os.path
import scipy.stats
import sys, traceback

if os.path.exists('phasedisp_minima.csv'):
    try:
        data = np.loadtxt('phasedisp_minima.csv')
        #print shape(data)
        if os.path.exists('phasedisp_bootstrap.csv'):
            try:
                bsData = np.loadtxt('phasedisp_bootstrap.csv')
                #print(shape(bsData))
                cohLenMin = 1
                cohLenMax = 1.64321608
                #if cohLenMax == cohLenMin:
                #   cohLenMax = data[1, 0]
                freq = data[1]
                closestFreqs = dict()
                for bsNo, bsCohLen, bsFreq, _, _ in bsData:
                    if (cohLenMin <= bsCohLen and cohLenMax >= bsCohLen and (not closestFreqs.has_key(bsNo) or abs(freq - bsFreq) < abs(freq - closestFreqs[bsNo]))):
                        closestFreqs[bsNo] = bsFreq
                #print(closestFreqs.values())
                bsFreqs = closestFreqs.values()

                #bsFreqs.append(freq)
                #if results_name == "bxmxy_0-42_64-128_30.0-70.0":
                hist(bsFreqs)
                (skewKurt, normality) = scipy.stats.normaltest(bsFreqs)
                freqStd = std(bsFreqs)
                if (freqStd/freq <= 0.1):
                    meanFreq = mean(bsFreqs)
                    #print("Normality prob: %f" % normality)
                    #print("cohLen freq: %f %f" % (cohLen, freq))
                    if (freq > meanFreq + freqStd or freq < meanFreq - freqStd):
                        print("%s Biased bootstrap sample" % (results_name))
                        #print ("There is a bias")
                    else:
                        print("%s %f %f %f %f %f" % (results_name, cohLenMax, freq, freqStd, normality, skewKurt))
                        #print freqStd
            except:
                #print sys.exc_info()[0]
                traceback.print_exc()
        else:
            print "ERROR: No minima file found"
    except:
        #print sys.exc_info()[0]
        traceback.print_exc()
else:
    print "ERROR: No minima file found"
