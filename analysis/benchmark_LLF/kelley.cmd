universe = vanilla 

getenv = true

executable = /usr/bin/Rscript

arguments = Generate_LLF_Causal_Predictions.R 400 450

#requirements = (TARGET.Machine == "isye-hpc0449.isye.gatech.edu")

#requirements = (Name == "isye-hpc1219.isye.gatech.edu")

log = $ENV(HOME)/Condor/$(Cluster).log

output = $ENV(HOME)/Condor/$(Cluster).$(process).out

error = $ENV(HOME)/Condor/$(cluster).$(Process).error

notification = error

notification = complete

notify_user = zwang937@gatech.edu

#request_memory = 40960

request_cpus = 60

queue
