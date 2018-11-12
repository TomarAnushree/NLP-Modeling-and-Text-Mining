#R version 3.4.3
load("review_sentences.RData")
review_sentences<-as.character(review_sentences)#100000 elements(reviews of all 3 data)
writeLines(review_sentences,con="review_sentence1.1.txt")

#R interface to to Google's Word2vec is rword2vec
#word2vec takes a text corpus as input and produce word vector as output.it construct a vocab from the training text data and then learns vector representation of words

#library(devtools)
#install_github("mukul13/rword2vec")
library(rword2vec)

#Training word2vec model for Semantic meaning
#To train text data to get word vectors:



model_sentence<-  word2vec(layer1_size = 300, train_file = "review_sentences1.1.txt", output_file = "vec_sent.bin",binary = 1, sample = 0.001, min_count = 40,num_threads=1,window=10)
#Vocab size: 18526 
#Words in train file: 16471312



#save model on disc

saveRDS(model_sentence,"./model_sentence1.1.rds")

#later...
model<-readRDS("./model_sentence1.1.rds")
print(model)


###convert .bin to .txt

bin_to_txt("vec_sent.bin","vector_sent1.1.txt")

data=as.data.frame(read.table("vector_sent1.1.txt",header = F,stringsAsFactors = F,skip=1,fill=TRUE))
colnames(data)[1]="word"

#Distance 
#To get closest words:
  
  ### file_name must be binary

dist=distance(file_name = "vec_sent.bin",search_word = "young",num = 10)
dist
### file_name must be binary
dist=distance(file_name = "vec_sent.bin",search_word = "thought",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_sent.bin",search_word = "queen",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_sent.bin",search_word = "awful",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_sent.bin",search_word = "nice",num = 10)
dist
### file_name must be binary
dist=distance(file_name = "vec_sent.bin",search_word = "king",num = 10)
dist

#Word analogy
#To get analogy or to observe strong regularities in the word vector space:
 
### file name must be binary
ana=word_analogy(file_name = "vec_sent.bin",search_words = "women child man kitchen",num = 10)
ana
 
  ### file name must be binary
  ana=word_analogy(file_name = "vec_sent.bin",search_words = "king queen man",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_sent.bin",search_words = "saw seen movie",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_sent.bin",search_words = "dragon doves chirpy",num = 10)
ana


#Training word2phrase model
#To convert words to phrases:
  
#word2phrase(train_file = "text1.txt",output_file = "vec_sent.txt")

### use this new text file to give word vectors
#model1=word2vec(train_file = "vec.txt",output_file = "vec2.bin",binary=1)
#saveRDS(model1,"./model1.rds")
#Word count 
#To do word count:
  
  ### to count word occurences in input file
vocab_count("review_sentences1.1.txt", "vocab_file.txt", verbose = 2, max_vocab = 0,
            min_count = 1)
#Counted 159798 unique words.
#Using vocabulary of size 159798.

library(readr)

df<-read_table("vocab_file.txt")
head(df)





























#Reference
#http://www.rpubs.com/mukul13/rword2vec
