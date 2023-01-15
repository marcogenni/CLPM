#!/usr/bin/env Rscript
rm(list=ls())
setwd("./")

el <- read.csv("ht09_contact_list.dat", sep = "\t", header = F)
colnames(el) = c("timestamp", "sender", "receiver")

el_min <- min(el[, 2:3])
el[,2] = el[,2] - el_min + 1
el[,3] = el[,3] - el_min + 1

n_nodes <- length(unique(c(el[,2], el[,3])))
permutation <- rep(NA, n_nodes)

index <- 1
for (l in 1:nrow(el)) for (k in 2:3)
{
  if (is.na(permutation[el[l,k]]))
  {
    permutation[el[l,k]] = index
    index = index + 1
  }
}

for (l in 1:nrow(el)) for (k in 2:3) el[l,k] = permutation[el[l,k]]
el_export <- el
el_export[,1] = el_export[,1] - min(el_export[,1])
el_export[,1] = el_export[,1] / 3600
el_export = el_export[el_export[,1] < 13,]

el_export[,2] = el_export[,2] - 1
el_export[,3] = el_export[,3] - 1

write.table(el_export, file = "edgelist.csv", sep = ",", row.names = F)
