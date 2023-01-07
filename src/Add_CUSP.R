list.of.packages <- c("ggplot2", "Rcpp", "grf", "caret", "mltools", "rpart", "minpack.lm", "doParallel", "rattle", "anytime")
list.of.packages <- c(list.of.packages, "plyr", "zoo","usmap","readxl","lubridate","tidyverse","data.table")


new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib='/home/zwang937/local/R_libs', repos="http://cran.us.r-project.org", dependencies = TRUE, INSTALL_opts = '--no-lock')
# Will need to add custom installation folder for servers without admin access
lapply(list.of.packages, require, character.only = TRUE) 






# Load CUSP Data


#CUSP = paste("../data/COVID-19 US state policy database (CUSP)",".xlsx",sep="")

CUSP = paste("../data/COVID-19 US state policy database 3_30_2022 imputted",".xlsx",sep="")

# Pre-processing CUSP data

#df <- read_excel(CUSP, sheet=2, n_max=51)

df <- readxl::read_excel(CUSP, sheet=1, skip=1, n_max=54)

names(df) <- gsub(" ","_",names(df))


#df[, which(df[3,i] %in% c("date"))]


for(i in 1:length(names(df))){
  print(df[3,i])
  if(df[3,i]=="date"){
    df[,i][df[,i]=="0"]<-"50000"
    df[,i]<-as.Date(as.numeric(unlist(df[,i])), origin="1899-12-30")
#    for(j in 4:54){
#      df[j,i]<-as.Date(as.numeric(df[j,i]), origin="1899-12-30")
#      df[j,i]<-as.numeric(df[j,i])
#      }
    }
}

df<-df[-c(1,2,3),]

#foreach(i = 3:13)%do%{ df <- merge(df, read_excel(CUSP, i, n_max=51)) }
#foreach(i = 15:18)%do%{ df <- merge(df, read_excel(CUSP, i, n_max=51)) }

#for(i in 4:length(names(df))){
  # If the data is not all binary, then it is a calendar column
#  if(!(all(df[,i]  %in% c(0,1), na.rm = TRUE))){
    # For some columns with dates, there are 0, we should have treated them as NA
    # Format them as 2030 instead
    # For sheet "State Characteristics", values are neither 0,1 nor calendar, we leave it alone
    # Only change those calendar columns
#    if(all(format(as.Date(df[,i], origin="1899-12-30"),"%Y")  %in% c(1899,2019,2020), na.rm = TRUE)){
#      df[,i]<-as.Date(df[,i], origin="1899-12-30")
#      for (j in 1:51) {if (year(df[j,i])==1899) {year(df[j,i])<-2030}}
#    } else {next}
#  }
#}



# DROP State
df <- df[, -which(names(df) %in% c("State"))]


print("Now merging CUSP and county_data_augmented")

#merge CUSP Data with county_data_augmented Data

destfile =  paste("../data/augmented_us-counties_latest",".csv",sep="")

county_data_augmented = as.data.frame(fread(file=destfile))

county_data_augmented["FIPS_Code"]<- as.numeric(fips(county_data_augmented$state, county = c()))


#data<-merge(x=county_data_augmented, y=df, by = "FIPS_Code", all.x = TRUE)
data <- join(x=county_data_augmented, y=df, by="FIPS_Code", type="left")

data<- data %>% dplyr::rename(State_FIPS_Code=FIPS_Code)

data$datetime<-as.Date(data$datetime, "%Y-%m-%d")

cnames = names(county_data_augmented)
rm(county_data_augmented)
rm(df)
gc()
print("Freed memory")
for (i in (length(cnames)+2):length(names(data))) {
  # Set number of days  policies that have already started by this datetime as 1,2,3...., 
  # otherwise 0
  if (inherits(data[,i], 'Date')){
    data[,i]<-data$datetime-data[,i]+1
    data[,i][data[,i]<0]<-0
    data[,i]<-as.numeric(data[,i])
  }
  if (inherits(data[,i], 'character')){
    data[,i]<-as.numeric(data[,i])
  }
}
gc()

print("Loading Tracking Data")
# URL of COVID Tracking Data
track_url<-"https://covidtracking.com/data/download/all-states-history.csv"
track_data<-as.data.frame(fread(track_url))

# Pre-processing the data

track_data<-track_data[, !(names(track_data) %in% c("dataQualityGrade"))]
track_data$date<-ymd(track_data$date)


#merge COVID Tracking Data with data
track_data <- track_data %>% dplyr::rename( State_Abbreviation=state)
print("Forming dataT")
#dataT<-merge(x=data, y=track_data, by = c("State_Abbreviation","date"), all.x = TRUE)
dataT <- join(x=data, y=track_data, b= c("State_Abbreviation","date"), type="left")
rm(data)
rm(track_data)
gc()
# DROP State_Abbreviation, state 
dataT <- dataT[, -which(names(dataT) %in% c("State_Abbreviation"))]

act_url<-"https://api.covidactnow.org/v2/counties.timeseries.csv?apiKey=e279791654264256bb7896d5a7b00e82"
act_data<-as.data.frame(fread(act_url))

act_data$date<-as.Date(act_data$date, "%Y-%m-%d")
act_data <- act_data[, which(names(act_data) %in% c("date","fips","metrics.testPositivityRatio","metrics.vaccinationsInitiatedRatio","metrics.vaccinationsCompletedRatio"))]
gc()


end<-max(dataT$date)
print("Writing dataT")
fwrite(dataT, "../data/dataT.csv", row.names=FALSE)
rm(dataT)
gc()


print("Forming act_data")

act_url<-"https://api.covidactnow.org/v2/counties.timeseries.csv?apiKey=e279791654264256bb7896d5a7b00e82"
act_data<-as.data.frame(fread(act_url))

act_data$date<-as.Date(act_data$date, "%Y-%m-%d")
act_data <- act_data[, which(names(act_data) %in% c("date","fips","metrics.testPositivityRatio","metrics.vaccinationsInitiatedRatio","metrics.vaccinationsCompletedRatio"))]
print("Forming act_dataF")
fips_list = sort(unique(act_data$fips))
act_dataF <- act_data[FALSE,] #clear all entry
for (fips in fips_list){

  fips.df <- act_data[which(act_data$fips==fips),]
  if (dim(fips.df)[1] == 0){
    print(paste("fips ",toString(fips)," has no entry ",sep=""))
    next
  }
  first.fips.date <- min(fips.df$date)
  last.fips.date <- max(fips.df$date)
  
  if(first.fips.date == last.fips.date){
    print(paste("fips ",toString(fips)," only has one entry ",sep=""))
    next
  }
  
  fips.df[which(fips.df$date==first.fips.date),"metrics.testPositivityRatio"]<-0
  fips.df[which(fips.df$date==first.fips.date),"metrics.vaccinationsInitiatedRatio"]<-0
  fips.df[which(fips.df$date==first.fips.date),"metrics.vaccinationsCompletedRatio"]<-0
  
  fips.df<-na.locf(fips.df) #Last Observation Carried Forward
  #fips.df$metrics.testPositivityRatio =  zoo::rollmean(fips.df$metrics.testPositivityRatio, 7, fill=NA, align="right")
  
  if(last.fips.date<end){
    imputter <- fips.df[which(fips.df$date==last.fips.date),]
    imputter$date<-end
    fips.df<-rbind(fips.df,imputter)
  }
  
  fips.df$metrics.vaccinationsCompletedRatio[fips.df$metrics.vaccinationsCompletedRatio>1] <- 1
  fips.df$metrics.vaccinationsInitiatedRatio[fips.df$metrics.vaccinationsInitiatedRatio>1] <- 1
  
  #fips.df[,which(dataF$metrics.vaccinationsCompletedRatio > 1) ] <- 1
  
  #fips.df %>% mutate(fips.df[,"metrics.vaccinationsCompletedRatio"] = ifelse(fips.df[,"metrics.vaccinationsCompletedRatio"] > 1, 1, fips.df[,"metrics.vaccinationsCompletedRatio"]),
   #                  fips.df[,"metrics.vaccinationsInitiatedRatio"] = ifelse(fips.df[,"metrics.vaccinationsInitiatedRatio"] > 1, 1, fips.df[,"metrics.vaccinationsInitiatedRatio"]))
  
  
  if(fips.df[which(fips.df$date>=end),"metrics.vaccinationsInitiatedRatio"]==0){
    fips.df[which(fips.df$date>=end),"metrics.vaccinationsInitiatedRatio"]<-NA
    fips.df[which(fips.df$date>=end),"metrics.vaccinationsCompletedRatio"]<-NA
  }
  
  #if(fips.df[,"metrics.vaccinationsInitiatedRatio"]>1){fips.df[,"metrics.vaccinationsInitiatedRatio"]<-1}
  #if(fips.df[,"metrics.vaccinationsCompletedRatio"]>1){fips.df[,"metrics.vaccinationsCompletedRatio"]<-1}
  
  
  # Append the data
  act_dataF<-rbind(act_dataF,fips.df)
  
  print(paste("fips ",toString(fips)," vaccination data complete ",sep=""))
}
rm(act_data)
rm(fips.df)
gc()

print("Writing act_dataF")

#fwrite(dataT, "../data/dataT.csv", row.names=FALSE)
fwrite(act_dataF, "../data/act_dataF.csv", row.names=FALSE)

closeAllConnections()
#dataF<-merge(x=dataT, y=act_dataF, by = c("fips","date"), all.x = TRUE)

#dataF<-join(x=dataT, y=act_dataF, by = c("fips","date"), type="left")
#rm(dataT)
#rm(act_dataF)
#gc()

#end_file = paste("../data/augmented_us-counties-states_latest",".csv",sep="")
#end_file = paste("../data/processed_us-counties_latest",".csv",sep="")


#fwrite(dataF, file=end_file, row.names=FALSE)


#closeAllConnections()

