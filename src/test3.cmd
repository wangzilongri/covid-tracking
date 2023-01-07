universe = parallel

getenv = true

executable = update_script.sh

requirements = (Machine == "isye-gpu1001.isye.gatech.edu")

#requirements = (Machine == "isye-hpc0443.isye.gatech.edu")

log = $ENV(HOME)/Condor/$(Cluster).log

output = $ENV(HOME)/Condor/$(Cluster).$(process).out

error = $ENV(HOME)/Condor/$(cluster).$(Process).error

notification = error

notification = complete

notify_user = zwang937@gatech.edu

request_memory = 409600

machine_count = 1

request_cpus = 16

queue
