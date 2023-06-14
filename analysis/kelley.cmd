universe = vanilla 

getenv = true

executable = individual_county_grf_condor.sh

#requirements = (Name == "isye-gpu1001.isye.gatech.edu")

#requirements = (Name == "isye-hpc1219.isye.gatech.edu")

log = $ENV(HOME)/Condor/$(Cluster).log

output = $ENV(HOME)/Condor/$(Cluster).$(process).out

error = $ENV(HOME)/Condor/$(cluster).$(Process).error

notification = error

notification = complete

notify_user = zwang937@gatech.edu

request_memory = 40960

#request_cpus = 60

queue
