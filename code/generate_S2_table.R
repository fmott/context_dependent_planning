rm(list = ls())
library(data.table)
library(corrplot)
library(lme4)
library(lmerTest)
library(optimx)
## Set current directory to source file location if run in Rstudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

###########################################################
# Testing for sequence effects 
###########################################################
## Load stuff 
filename <- Sys.glob(file.path("../data/behaviour/data_all_participants_20220215120148.csv"))
dat <- fread(filename)

### Prepare data
idx <- (dat[,index] >= 4) & (dat[,index] <= 239)
response  <- dat[idx,response] == 0 # 1 = accept, 0 = wait
segment <-  dat[idx,segment] == 0 # 1 = LC, 0 = HC
timeout <- dat[idx,timeout]
participant <- dat[idx,vpn]

idx <- (dat[,index] >= 0) & (dat[,index] <= 235)
segment_previous = dat[idx,segment] == 0

# remove timeout trials 
idx <- (timeout==0)
response <- response[idx]
segment <- segment[idx]
segment_previous <- segment_previous[idx]
participant <- participant[idx]

## Estimate model 

res <- glmer(formula = response ~  segment + segment_previous + segment*segment_previous  + (1 +  segment + segment_previous + segment*segment_previous| participant),
             family = binomial(link = "logit"),control = glmerControl(optimizer ='optimx', optCtrl=list(method='nlminb')))

## Display results
summary(res)