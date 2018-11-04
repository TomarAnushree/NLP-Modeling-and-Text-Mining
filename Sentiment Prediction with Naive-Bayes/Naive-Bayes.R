#set working dir. where data is

#Read labeledTrainData data 

library(readr)
L_traindata <- read_delim("labeledTrainData.tsv", 
                          "\t", escape_double = FALSE, trim_ws = TRUE)

View(L_traindata)
dim(L_traindata)
colnames(L_traindata)
head(L_traindata)
gettext(L_traindata[1,3])
L_traindata$sentiment<-as.factor(L_traindata$sentiment)


#Read unlabeled data
testData <- read_delim("testData.tsv","\t", escape_double = FALSE, trim_ws = TRUE)
dim(testData)
colnames(testData) 
head(testData)
gettext(testData[4,2])

#Data Cleansing and Text Preprocessing 

#row bind the review of test and traindata for text processing 

review_train_test<-rbind(L_traindata[1:4000,3],testData[1:1000,2])

#Create Label
label<-L_traindata[1:4000,2]
str(label)

#Removing HTML tags
review_train_test["review"]<-as.data.frame(gsub("<br /><br />"," ",review_train_test$review))
gettext(review_train_test$review[1])
gettext(review_train_test$review[4001])

#gsub used for Pattern Matching and Replacement in a string

library(NLP)
library(tm)#require for transformation carpora
library(SnowballC)#for stemming


# Prepare corpus for the text data which is a collection of documents containing text.
#a vector source interprets each element of the vector x as a document

review_corpous<-Corpus(VectorSource(review_train_test$review))

# Cleaning data (removing unwanted symbols)
corpus_clean<-tm_map(review_corpous,tolower)
corpus_clean<-tm_map(corpus_clean, removeNumbers)
corpus_clean<-tm_map(corpus_clean,removeWords, stopwords())
corpus_clean<-tm_map(corpus_clean,removePunctuation)
corpus_clean<-tm_map(corpus_clean,stemDocument)
corpus_clean<-tm_map(corpus_clean,stripWhitespace)
class(corpus_clean)
inspect(corpus_clean[1])#train data
inspect(corpus_clean[4001])#test data

# create a document-term sparse matrix
review_dtm <- DocumentTermMatrix(corpus_clean) 
class(review_dtm)


# creating training, test and validation  datasets

dtm_train <- review_dtm[1:3000,]
dtm_val<-review_dtm[3001:4000,]
dtm_test  <- review_dtm[4001:5000,]#new data unlabled

corpus_train <- corpus_clean[1:3000]
corpus_val<-corpus_clean[3001:4000]
corpus_test  <- corpus_clean[4001:5000]


# indicator features for frequent words
review_dict<-findFreqTerms(dtm_train, 5)


traindtm <- DocumentTermMatrix(corpus_train, list(dictionary = review_dict))
valdtm<-DocumentTermMatrix(corpus_val,list(dictionary=review_dict))
                                           
testdtm  <- DocumentTermMatrix(corpus_test, list(dictionary = review_dict))
                                                 



dim(traindtm)
dim(testdtm)
dim(valdtm)
class(traindtm)
inspect(corpus_train[1])
inspect(corpus_val[1])
inspect(corpus_test[1])
list(review_dict[1:100])
inspect(testdtm[1:5,1:20])
inspect(traindtm[1:5,1:20])
inspect(valdtm[1:5,1:20])


# convert counts to a factor for to process through NB,ML algorithm.

convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("Absent", "Present"))
}

# apply() convert_counts() to columns of train/test data
train <- apply(traindtm, MARGIN = 2, convert_counts)
val  <- apply(valdtm, MARGIN = 2, convert_counts)
test<-apply(testdtm, MARGIN = 2, convert_counts)
View(train)
View(val)
View(test)

##  Training a model on the data ----

########################### #Naive Bayes Classification# ############################

library(e1071)
NB_classifier <- naiveBayes(train, label$sentiment[1:3000])
NB_classifier

##  Evaluating model performance ----
val_pred <- predict(NB_classifier, val)

# Classification Accuracy - num. of correct pred/total no. of pred.
mean(val_pred==label$sentiment[3001:4000])# 0.818

library(caret)
confusionMatrix(table(val_pred,label$sentiment[3001:4000]))



NB_classifier2 <- naiveBayes(train, label$sentiment[1:3000], laplace = 1)
val_pred2 <- predict(NB_classifier2, val)

# Classification Accuracy 
mean(val_pred2==label$sentiment[3001:4000])#0.828

#Classification Accuracy is not enough to select a model
confusionMatrix(table(val_pred2,label$sentiment[3001:4000]))


#if there is large class imbalance, model can predict the value of the mejority class for all predictions
#and achieve a high classification accuracy this is called "Accuracy Paradox".we need additional measures to
#evaluate a classifier.


#F1-Score 2*((precision*recall)/precision+recall))
#Precision is the num of TP/ TP + FP,low Precision means large num of FP.
#recall= TP/TP+FN called senstivity or True positive rate, low recall means more FN.
#F1-score conveys the balance between the precision and the recall.

library(MLmetrics)
F1_Score(label$sentiment[3001:4000],val_pred2,positive = NULL)#0.8349328

#Prediction on new unlabelled data
test_pred<- predict(NB_classifier2, test)
test_pred<-as.data.frame(test_pred)
test_pred<-cbind(testData[1:1000,1],test_pred)
colnames(test_pred)[1]<-paste("id")
colnames(test_pred)[2]<-paste("sentiment")

write.csv(test_pred,"test_pred.csv",row.names = FALSE)

#####################################################################################


#referrence:
#https://machinelearningmastery.com/classification-accuracy-is not-enough-more-performance-measures-you-can--use/
