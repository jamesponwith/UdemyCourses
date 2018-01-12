# Data Preprocessing

# Importing data set
dataset = read.csv('Data.csv')
# dataset = dataset[, 2:3]

# Splitting dataset into Training set and Testing set
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) # input % towards training
training_set = subset(dataset, split == TRUE)
testing_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# testing_set[, 2:3] = scale(testing_set[, 2:3])


