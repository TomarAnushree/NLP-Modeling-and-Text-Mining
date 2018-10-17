#R version 3.5.1

library(text2vec)
library(magrittr)
library(data.table)
library(readr)

#Read labeledTrainData data 

traindata <- read_delim("labeledTrainData.tsv", 
                          "\t", escape_double = FALSE, trim_ws = TRUE)
traindata<-data.table(traindata)

#Read UnlabeledTrainData data as

newdata <- read_delim("testData.tsv", 
                        "\t", escape_double = FALSE, trim_ws = TRUE)
newdata<-data.table(newdata)

#splitting of traindata into test and train
setkey(traindata, id)
set.seed(2017L)
all_ids = traindata$id
train_ids = sample(all_ids, 20000)
test_ids = setdiff(all_ids, train_ids)
train<-traindata[J(train_ids)]
test<-traindata[J(test_ids)]


#Vectorization:create a vocabulary-based DTM.

# define preprocessing function and tokenization function
#create an iterator over tokens with the itoken() function and a vacabulary with create_vocabulary()function

prep_fun = tolower
tok_fun = word_tokenizer

it_train = itoken(train$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$id, 
                  progressbar = FALSE)
vocab = create_vocabulary(it_train)
vocab

#create dtm
vectorizer = vocab_vectorizer(vocab)
t1 = Sys.time()
dtm_train = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))
dim(dtm_train)
colnames(dtm_train)
identical(rownames(dtm_train), train$id)

# fit a logistic regression model with an L1 penalty and 4 fold cross-validation.
library(glmnet)
NFOLDS = 4
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 4-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))



#Process test data using same preprocessing,tokenization and vectorization function 

it_test = itoken(test$review, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = test$id, 
                  progressbar = FALSE)

#create dtm
dtm_test = create_dtm(it_test, vectorizer)
dim(dtm_test)
colnames(dtm_test)
identical(rownames(dtm_test), test$id)

##Evaluate Model performance
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)

#performance on the test data is roughly the same as we expect from cross-validation.
###################################################################################

#Pruning vocabulary
#remove pre-defined stopwords, very common and very unusual terms.

stopwords<- stopwords()
t1 = Sys.time()
vocab = create_vocabulary(it_train, stopwords = stopwords)
print(difftime(Sys.time(), t1, units = 'sec'))

pruned_vocab = prune_vocabulary(vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5,
                                doc_proportion_min = 0.001)
vectorizer = vocab_vectorizer(pruned_vocab)

# create dtm_train with new pruned vocabulary vectorizer
t1 = Sys.time()
dtm_train  = create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

dim(dtm_train)

# fit a logistic regression model with an L1 penalty and 4 fold cross-validation.
library(glmnet)
NFOLDS = 4
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              # L1 penalty
                              alpha = 1,
                              # interested in the area under ROC curve
                              type.measure = "auc",
                              # 4-fold cross-validation
                              nfolds = NFOLDS,
                              # high value is less accurate, but has faster training
                              thresh = 1e-3,
                              # again lower number of iterations for faster training
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

#create dtm_test with same vectorizer
dtm_test = create_dtm(it_test, vectorizer)
dim(dtm_test)

##Evaluate Model performance
preds = predict(glmnet_classifier, dtm_test, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)


#######################################################################################

#Improve model using N-grams instead of terms

t1 = Sys.time()
vocab = create_vocabulary(it_train, ngram = c(1L, 2L))
print(difftime(Sys.time(), t1, units = 'sec'))

vocab = prune_vocabulary(vocab, term_count_min = 10, 
                         doc_proportion_max = 0.5)

bigram_vectorizer = vocab_vectorizer(vocab)

dtm_train = create_dtm(it_train, bigram_vectorizer)

t1 = Sys.time()

glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 4,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
summary(glmnet_classifier)
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))#0.9557

#Evaluate Model performance
# apply vectorizer
dtm_test = create_dtm(it_test, bigram_vectorizer)
preds = predict.cv.glmnet(glmnet_classifier, dtm_test, type = 'class')[,1]
glmnet:::auc(test$sentiment, preds)#0.89

table(train$sentiment)

#if there is large class imbalance, model can predict the value of the mejority class for all predictions
#and achieve a high classification accuracy this is called "Accuracy Paradox".we need additional measures to
#evaluate a classifier.


#F1-Score 2*((precision*recall)/precision+recall))
#Precision is the num of TP/ TP + FP,low Precision means large num of FP.
#recall= TP/TP+FN called senstivity or True positive rate, low recall means more FN.
#F1-score conveys the balance between the precision and the recall.

library(MLmetrics)
F1_Score(test$sentiment,preds,positive = NULL)#0.88


#Process newdata using same preprocessing,tokenization and vectorization function then make
#pridiction using best model

it_newtest = itoken(newdata$review, 
                    preprocessor = prep_fun, 
                    tokenizer = tok_fun, 
                    ids = newdata$id, 
                    progressbar = FALSE)


dtm_newtest = create_dtm(it_newtest, bigram_vectorizer)
predsnew = predict(glmnet_classifier, dtm_newtest, type = 'class')[,1]
predsnew<-as.data.frame(predsnew)
predsnew["id"]<-row.names(predsnew)
names(predsnew)[1]<-paste("sentiments")
write.csv(predsnew,"sentimentsTestData.csv",row.names = FALSE)
####################################################################################

#Feature Hashing
#fast and space-efficient way of vectorizing features, i.e. turning arbitrary features into indices in a vector or matrix.
#It works by applying a hash function to the features and using their hash values as indices directly,

h_vectorizer = hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L, 2L))

t1 = Sys.time()
dtm_train = create_dtm(it_train, h_vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 4,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))

dtm_test = create_dtm(it_test, h_vectorizer)

preds1 = predict(glmnet_classifier, dtm_test , type = 'response')[, 1]
glmnet:::auc(test$sentiment, preds1)

#AUC is not good but we can apply this on large douments.
###################################################################################
#Basic transformations
#when length of the documents vary we can apply "L1" normalization. It means we will transform rows in a way that sum of the row values will be equal to 1

#Normalization

dtm_train_l1_norm = normalize(dtm_train, "l1")
 
#TF-IDF
#It will not only normalize DTM, but also increase the weight of terms which are specific to a single document or 
#handful of documents and decrease the weight for terms used in most documents:

vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tfidf model
tfidf = TfIdf$new()
# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf = create_dtm(it_test, vectorizer)
dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)

#fit model on TF-IDF
t1 = Sys.time()
glmnet_classifier = cv.glmnet(x = dtm_train_tfidf, y = train[['sentiment']], 
                              family = 'binomial', 
                              alpha = 1,
                              type.measure = "auc",
                              nfolds = 5,
                              thresh = 1e-3,
                              maxit = 1e3)
print(difftime(Sys.time(), t1, units = 'sec'))
plot(glmnet_classifier)
print(paste("max AUC =", round(max(glmnet_classifier$cvm), 4)))
preds = predict(glmnet_classifier, dtm_test_tfidf, type = 'response')[,1]
glmnet:::auc(test$sentiment, preds)
############################################################################














#Reference:
#https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html
#https://machinelearningmastery.com/classification-accuracy-is-not-enough-more-performance-measures-you-can--use/
