#!/bin/bash
Rscript Individual_County_State_Forests_condor.R 2>&1 | tee ~/logs/individual_county_grf_condortxt
