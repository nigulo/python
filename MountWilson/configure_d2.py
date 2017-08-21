import sys
import os
import subprocess
import numpy as np

assert(len(sys.argv) >= 1)

bootstrapSize = 0
if (len(sys.argv) > 1):
    bootstrapSize = int(sys.argv[1])

selected_star=None
if (len(sys.argv) > 2):
    selected_star = sys.argv[2]

###########################################################################
# Create global run.sh
runSh = open("run.sh", 'w')

for root, dirs, files in os.walk("detrended"):
    
    for file in files:
        if file[-4:] == ".dat":
            star = file[:-4]
            if (selected_star != None and selected_star != star):
                continue
            print "Configuring " + star
    
            data = np.loadtxt("detrended/" + file)
            duration = data[-1,0] - data[0,0]
    
            param_file_name = "d2_params/parameters_" + star + ".txt"
            runSh.write("./D2 " + param_file_name + "\n")
            
            tscale = 1
    
            parametersTxt = open(param_file_name, 'w')
            parametersTxt.write(
                "binary=0\n"
                + "phaseSelFn=cosine\n"
                + "timeSelFn=none\n"
                + "bufferSize=10000\n"
                + "dims=1\n"
                + "numProcs=1\n"
                + "regions=0-0\n"
                + "numVars=1\n"
                + "varIndices=0\n"
                + "minPeriod=2\n"
                + "maxPeriod=" + str(round(duration * tscale / 1.5)) + "\n"
                + "tScale=" + str(tscale) + "\n"
                + "bootstrapSize=" + str(bootstrapSize) +" \n"
                + "varScales=1\n"
                + "filePath=detrended/" + file + "\n"
                + "outputFilePath=d2_res/" + star + "\n")
            parametersTxt.close()

    runSh.close()
        
p = subprocess.Popen('chmod +x run.sh', shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
retval = p.wait()        
