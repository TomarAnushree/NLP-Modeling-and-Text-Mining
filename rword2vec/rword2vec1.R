#R version 3.4.3

#R interface to to Google's Word2vec is rword2vec
#word2vec takes a text corpus as input and produce word vector as output.it construct a vocab from the training text data and then learns vector representation of words

#library(devtools)
#install_github("mukul13/rword2vec")
library(rword2vec)

#Training word2vec model for Semantic meaning
#To train text data to get word vectors:

model<-  word2vec(layer1_size = 300, train_file = "text8", output_file = "vec_text8.bin",binary = 1, sample = 0.001, min_count = 40,num_threads=1)

#Vocab size: 21149
#Words in train file: 16056242


###convert .bin to .txt
bin_to_txt("vec_text8.bin","vector_text8.txt")
data=as.data.frame(read.table("vector_text8.txt",header = F,stringsAsFactors = F,skip=1))
colnames(data)[1]="word"
print(str(data))


#Distance 
#To get closest words:

### file_name must be binary

dist=distance(file_name = "vec_text8.bin",search_word = "king",num = 10)
dist
### file_name must be binary
dist=distance(file_name = "vec_text8.bin",search_word = "terrible",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_text8.bin",search_word = "queen",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_text8.bin",search_word = "awful",num = 10)
dist

### file_name must be binary
dist=distance(file_name = "vec_text8.bin",search_word = "nice",num = 10)
dist

#Word analogy
#To get analogy or to observe strong regularities in the word vector space:

### file name must be binary
ana=word_analogy(file_name = "vec_text8.bin",search_words = "women child man kitchen",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_text8.bin",search_words = "king queen man",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_text8.bin",search_words = "paris france berlin",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_text8.bin",search_words = "france england germani berlin",num = 10)
ana

### file name must be binary
ana=word_analogy(file_name = "vec_text8.bin",search_words = "paris berlin london austria",num = 10)
ana

#Training word2phrase model
#To convert words to phrases:

#model=word2phrase("text8","vec_text8.txt")

### use this new text file to give word vectors
#model1=word2vec(train_file = "vec_text8.txt",output_file = "vec_text8.1.bin",binary=1)

#Word count 
#To do word count:

### to count word occurences in input file
data<-vocab_count("text8", "vocab_text8.txt")
  
         
#Counted 253854 unique words.
#Using vocabulary of size 253854.

df<-read.table("vocab_text8.txt")
head(df)

























#Reference
#http://www.rpubs.com/mukul13/rword2vec
##https://stackoverflow.com/questions/30901595/word2vec-sentiment-classification-with-r-and-h2o
####https://github.com/mukul13/Kaggle---Bag-of-Words-Meets-Bag-of-Popcorns-using-Word2vec-in-R