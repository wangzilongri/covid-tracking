universe = vanilla 

getenv = true

executable = individual_county_grf.sh

#requirements = (Machine == "isye-gpu1001.isye.gatech.edu")

#requirements = (Machine == "isye-hpc0443.isye.gatech.edu")

log = $ENV(HOME)/Condor/$(Cluster).log

output = $ENV(HOME)/Condor/$(Cluster).$(process).out

error = $ENV(HOME)/Condor/$(cluster).$(Process).error

notification = error

notification = complete

notify_user = zwang937@gatech.edu

request_memory = 40960

request_cpus = 16

queue
