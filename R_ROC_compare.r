#install.packages("pROC")

library(pROC)


setwd("C:\\Users\\Administrator\\PycharmProjects\\d2d-interactions\\results\\holdout")
delim = ","  
dec = "."    
GT1 <- scan("GT1.txt", what="", sep="\n") #they should be equal
GT2 <- scan("GT2.txt", what="", sep="\n") #they should be equal
GT3 <- scan("GT3.txt", what="", sep="\n") #they should be equal
pred_nn <- read.csv("pred_nn.txt", header=FALSE, sep=delim, dec=dec, stringsAsFactors=FALSE)$V1
pred_j <- read.csv("pred_j.txt", header=FALSE, sep=delim, dec=dec, stringsAsFactors=FALSE)$V1
pred_r <- read.csv("pred_r.txt", header=FALSE, sep=delim, dec=dec, stringsAsFactors=FALSE)$V1
#pred_nn <- scan("pred_nn.txt", what="", sep="\n")
#pred_j <- scan("pred_j.txt", what="", sep="\n")

roc1 <- roc(GT2, pred_nn)
roc2 <- roc(GT2, pred_j)
roc3 <- roc(GT2, pred_r)
roc.test(roc1, roc2, paired=TRUE)
roc.test(roc1, roc3, paired=TRUE)
#roc.test(roc1, roc2, method="bootstrap", boot.n=100,paired=TRUE)