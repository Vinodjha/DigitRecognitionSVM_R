library(caret)
library(kernlab)
library(readr)
library(deplyr)
library(gridExtra)
library(ggplot2)
library(doParallel)

#Reading the data for training and testing
setwd("C:/Users/user/Desktop/DA/data modling/Support vector machines/SVM Dataset")

data<-read.csv("mnist_train.csv")
test<-read.csv("mnist_test.csv")
set.seed(100)
train_indices<-sample(1:nrow(data),size=10000) # chosing 10000 random rows from actual training data
#for training purpose. (accuracy may vary slightly, as the seed was not set earlier, hence different sample might result in slighlt different accuracy )

# creating training data
train<-data[train_indices,]

# Data cleaning steps 
sum(is.na(train)) #no NA values in training dataset
# checking for any duplicate entry
sum(duplicated(train)) # returns 0, no duplicate entry
sum(is.na(test))#no NA values in testng dataset
#removing irrelelvent columns(which have value =0 for all its rows in both the training and testing)
colnames(train)[1]="digit"
#colnames(test)[1]="digit"
colnames(test)=colnames(train)
# checking outliers or invalid entries
sum(train>255) #0 
sum(train<0)#0 so all values lies between 0 and 255 which is right about an 8 bit image(this can be done using summary as well)
train_bckup1<-train
#converting all the non integral values to integer(if any)
train<-as.data.frame(sapply(train, as.integer))


#in a gray scale image the pixel value ranges from 0 to 255 and is always an integer. it is kind of a 
#matrix with values between 0 and 255. White denotes 255 and black denotes 0. In the dataset the background
#of the dataset images are black and numbers are written in white. So there will be pixel locations which always
#represent background for all the digits in every representation of testing and training images. So such values are always zero and can be easily
#removed to reduce complexity. Combining training and testing and removing the columns which are always zero.
t1<-rbind(train,test) # combining training and testing data just to find columns with all the zero values
t1_bckup1<-t1

x=colSums(t1)
redundant_indices<-which(x==0) # indices of the columns for which all the values are zero in t1
t1<-t1[,-x]
train<-t1[1:nrow(train),] # separating training and testing again
test<-t1[10001:19999,]
train$digit<-as.factor(train$digit)
summary(train$digit) # they seem to be almost in equal proportion , so our train data look balanced


#model building
# Linear model
model_linear <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "vanilladot")
training_predict<-predict(model_linear,train)
confusionMatrix(training_predict,train$digit) # 100 percent accuracy, as expected

training_predict<-predict(model_linear,test)
confusionMatrix(training_predict,test$digit) # 90 % ACCURACY

# RBF model
model_rbf <- ksvm(digit~ ., data = train, scale = FALSE, kernel = "rbfdot")
test_predict<-predict(model_rbf,test)
confusionMatrix(test_predict,test$digit) # 95.7% accuracy, 5 % higher than linear model,
 #which is not very high and we can keep our linear model as the default model for classification
#keeping the model simple and more more generalizable. Further we will tune the cost parameter of our linear model to see 
#if we can achieve better accuracy 
#switching parallel procesisng on
registerDoParallel(makeCluster(detectCores()-1))

#tuning the parameters of linear spoort vector classifier by cross validation
traincontrol<-trainControl(method="CV",number=10) # using cross validation 
metric <- "Accuracy" # using accuracy as our evaluation matric for getting best model
grid <- expand.grid(C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,5))
set.seed(100)

svm_Linear_Grid <- train(digit ~., data = train, method = "svmLinear",
                         trControl=traincontrol,
                         preProcess = c("center", "scale"),
                         tuneGrid = grid,
                         tuneLength = 10)
#Accuracy was used to select the optimal model using the largest value.

print(svm_Linear_Grid) # which indicates we can achieve accuracy as high as 92.3 % with our linear model
#when C=0.01.
plot(svm_Linear_Grid)
#checking the accuracy on testing data
predicted_digit<-predict(svm_Linear_Grid,test)
confusionMatrix(predicted_digit,test$digit) # overall accuracy is 93% which is almost same as the accuracy over 
#training data.This result is at par with the highest accuracy achieved in this dataset. This model is high on accuracy, less complex and generaizable
# A nonlinear model can similarly be tuned with its hyperparameters. e.g. an RBF model can be made
#more accurate by finding best fir hyperparameters sigma and C, by changing the method to svmRadial
#and changing the grid parameter so as to accomodate various values of sigma as well with various values of C.

#svm_Linear_Grid is my final model

# cross validating an RBF model
set.seed(100)
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                     0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                           C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                 1, 1.5, 2,5))
svm_Radial_Grid <- train(digit~., data = train, method = "svmRadial",
                           trControl=traincontrol, preProcess = c("center", "scale"),
                           tuneGrid = grid_radial,tuneLength = 10)

