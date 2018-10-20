#R version 3.4.3
#Exploratory Analysis of Text and Document Summarization of the corpus elements

library(readr)
traindata <- read_delim("labeledTrainData.tsv", 
                           "\t", escape_double = FALSE, trim_ws = TRUE)


#Remove HTML tags

traindata["review"]<-as.data.frame(gsub("<br /><br />"," ",traindata$review))

library(tidyverse)
library(tokenizers)

#Tokenization of Word from  reviews
# it will remove all the panctuation and convert everything into lowercase.

traindata$review<-as.character(traindata$review)
reviews_words<-tokenize_words(traindata$review[1:5])
length(reviews_words)#it gives a list of object or total number of reviews
length(reviews_words[[1]])#number of words in 1st review[430 words]

#data frame giving total words per review
total_words<-as.data.frame(sapply(reviews_words,length))

#cerate a data frame of frequency of words per document

freq<-as.data.frame(table(reviews_words[[1]]))
colnames(freq)[1]<-paste("word")
colnames(freq)[2]<-paste("count")
library(plyr)
freq<-arrange(freq,desc(count))
freq$word[1:5]
head(freq$word)
head(freq)

#remove commonly used words that are not giving insight of content of reviews.

usefulwords<-with(freq, subset(word, count < 5))

#Split each reviews into sentences
review_sentences<-tokenize_sentences(traindata$review)
review_sentences[[1]]#19 sentences
count_sentences(traindata1$review[1])
count_characters(traindata1$review[1])
count_words(traindata1$review[1])#430

#split sentences of each reviews into words so that we identify,how many words are present in each sentence of the reviews. 
tokenize_words(review_sentences[[1]])
sentence_2_word<-tokenize_words(as.character(review_sentences))
length(sentence_2_word)#25000
print(sentence_2_word[[1]])#430
print(sentence_2_word[[2]])#159
str(sentence_2_word)

#save an object of word list for future use
save(sentence_2_word,file = "sentence_2_word.RData")
save(review_sentences,file = "review_sentences.RData")
#later
load("sentence_2_word.RData")


#we need to apply above function to each reviews so create a for loop and save the results as an element of the vector description
#Document Summarization


library(dplyr)
description<-c()
for(i in 1:length(sentence_2_word)){
  tab<-table(sentence_2_word[[i]])
  tab<-data_frame(word=names(tab),count=as.numeric(tab))
  tab<-arrange(tab,desc(count))
  tab<-with(tab,subset(word,count<5))
  result<-head(tab)
  description<-c(description,paste(result,collapse = ";"))
}
cat(description,sep="\n")














