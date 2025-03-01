# setting directory
setwd("D:/Documents/kalej/Graduate School/2nd sem AY 2425/AMAT 255 - Mathematical Data Science/AM255 - Program Files and Codes")


# importing the data
da = read.csv('DOH COVID Data Drop_ 20230311 - 04 Case Information_batch_0.csv')
db = read.csv('DOH COVID Data Drop_ 20230311 - 04 Case Information_batch_1.csv')
dc = read.csv('DOH COVID Data Drop_ 20230311 - 04 Case Information_batch_2.csv')
dd = read.csv('DOH COVID Data Drop_ 20230311 - 04 Case Information_batch_3.csv')
de = read.csv('DOH COVID Data Drop_ 20230311 - 04 Case Information_batch_4.csv')

# concatenating dataset
all <- rbind(da, db, dc, dd, de)
colnames(all, 1)

# select specific columns
all_subset <- all[c(4, 7, 13)]  # column numbers to be retained
colnames(all_subset, 1)

# change sex type
all_subset[all_subset=="MALE"] = "M"
all_subset[all_subset=="FEMALE"] = "F"

all_subset = all_subset[-c(1)] #colnames in "all" is different from "all_subset"

# remove rows with missing values in ProvRes and DateRepConf
all_subset = subset(all_subset, ProvRes != "")
all_subset = subset(all_subset, DateRepConf != "")

all_subset = all_subset[-c(3500000:4077757),] #delete rows

# export datasets to csv files
write.csv(all, file='all.csv', row.names=FALSE)
write.csv(all_subset, file='all_subset.csv', row.names=FALSE)
