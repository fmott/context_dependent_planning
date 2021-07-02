rm(list = ls())
library(data.table)

###########################################################
# Logistic regression pooled over subjects. 
# Responses fitted against task features in basic trials.
###########################################################

## Load data 
filename <- Sys.glob(file.path("../data/behaviour/data_all_partipants_20210623102746.csv"))
dat <- fread(filename)

## Prepare data
idx <-  (dat[,timeout] == 0)  &  (dat[,is_basic] == 1)
response  <- dat[idx,response] == 0 # 1 = accept, 0 = wait
energy <- dat[idx,energy]
segment <-  dat[idx,segment] == 0 # 1 = LC, 0 = HC
segment_next <- dat[idx,segment_after] == 0 # 1 = LC, 0 = HC
trial <- dat[idx,trial]
reward <- dat[idx,reward]

## Estimate GLM
res <- glm(response ~ reward + energy + segment + segment_next + trial,family = binomial(link = "logit" ))

# Display summary
summary(res)
output <- as.data.frame(summary(res)$coefficients)

