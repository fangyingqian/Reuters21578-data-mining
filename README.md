# Reusters21578-data-mining
INTRODUCTION
--------------
This project is based on the original data set reuters21578. The data set consists totally 21578 documents, split into 21 SGML files. Some documents are assigned for training purpose, some are for testing and the others are not used. Totally 135 columns topic names are included and the final column is the document text, from which we can extract features. For each document, if the value for a particular topic name is ‘1’, then the document is assigned to this particular topic. This report adapted a strategy to assign each document with only one topic. In this report, Reuters21578 data set was firstly conducted pre-processing to obtain the features representation of documents, and then new dataset with features and topics was used for classification with two different classifiers. Finally, three clustering algorithms were applied based on the best performing feature.

EXAMPLE TO RUN THE CODE
------------
First of all, rename the original data as reut, and then just follow the codes:

text<-reut[,140]
install.packages("tm")
library(tm)
reuters<- Corpus(VectorSource(text)) #creat a corpus
install.packages("SnowballC")
library(SnowballC)

pre-processing cleaning
----------
reuters<-tm_map(reuters,removeNumbers) #remove numbers
reuters <- tm_map(reuters, content_transformer(tolower))#lower case  
reuters<-tm_map(reuters,removePunctuation) # remove punctuation
reuters <- tm_map(reuters, removeWords, stopwords("english")) # remove stop words
reuters <- tm_map(reuters, stripWhitespace)#remove extra whitespace
reuters<-tm_map(reuters, stemDocument)  #stemming
inspect(reuters)

Feature Engineering
---------------------
dtm <- DocumentTermMatrix(reuters);
inspect(dtm[25:30, 1:5])
to omit terms with low frequency
dtm1<-removeSparseTerms(dtm,sparse=0.95)
data<-as.data.frame(inspect(dtm1))
not.zero<-rowSums(data)!=0
newdata<-data[rowSums(data)>0,] # delete the documents with zero words

install.packages("topicmodels")
library(topicmodels)

find.k<-function(x){
  loglike<-vector()
  perp<-vector()
  for (k in 1:10){
  lda<-LDA(x,10*k)
  loglike[k]<-logLik(lda)
  }
  return(loglike)
}
find.k(newdata)
this function spent too much time to run
it seems like bigger k give bigger loglikelihood value, 
i chose k=10 by considering the work of computation 

when k=10
lda<-LDA(newdata,10)
lda.inf<-posterior(lda)$topics #alternative way, use lda@gamma to get the posterior
newdata2<-as.data.frame(lda.inf) 

newdata3<-cbind(newdata,newdata2)

assign each document with only one topic
col.sums<-colSums(reut[,4:138])
order<-order(col.sums)
top.ten<-tail(order,10)
col.sums[top.ten]
reut1<-subset(reut,not.zero)   delete the documents with zero words from the original dataset
reut2<-reut1[,top.ten+3] reduce the data with only the top 10 topics
not.zero1<-rowSums(reut2)!=0
reut3<-subset(reut2,not.zero1)  find the documents which belongs to the top 10 topics
    
to creat a function to assign each document with only one topic
topics<-names(reut2) #top 10 topics
assign.topic<-function(x){
  final.topic<-vector()
  for (i in 1:nrow(x)){
    which.is.one<-x[i,]==1
    set.seed=75
    final.topic[i]<-sample(topics[which.is.one],1)
  }
  return(final.topic)
}
final.topics<-assign.topic(reut3)

 obtain the final data for classification
feature1<-subset(newdata,not.zero1)
feature2<-subset(newdata2,not.zero1)
feature3<-subset(newdata3,not.zero1)
data.feature1<-cbind(feature1,final.topics)
data.feature2<-cbind(feature2,final.topics)
data.feature3<-cbind(feature3,final.topics)


Classification
----------------

NB
---
install.packages('e1071')
library(e1071)
accuracy.nb<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  accuracy<-vector()
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    nb1<-naiveBayes(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.nb1<-predict(nb1,testset[,1:(ncol(data)-2)])
    table.nb<-table(pre.nb1,testset[,(ncol(data)-1)])
    acc<-sum(diag(table.nb))/sum(table.nb)
    accuracy[i]<-round(acc,4)
  }
  accuracy[k+1]<-mean(accuracy)
  accuracy[k+2]<-sd(accuracy)
  return(accuracy)  
}

naivebayes1<-accuracy.nb(data.feature1,10);naivebayes1
naivebayes2<-accuracy.nb(data.feature2,10);naivebayes2
naivebayes3<-accuracy.nb(data.feature3,10);naivebayes3

SVM
---
accuracy.svm<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  accuracy<-vector()
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    svm1<-svm(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.svm1<-predict(svm1,testset[,1:(ncol(data)-2)])
    table.svm<-table(pre.svm1,testset[,(ncol(data)-1)])
    acc<-sum(diag(table.svm))/sum(table.svm)
    accuracy[i]<-round(acc,4)
  }
  accuracy[k+1]<-mean(accuracy)
  accuracy[k+2]<-sd(accuracy)
  return(accuracy)  
}

SVM1<-accuracy.svm(data.feature1,10);SVM1
SVM2<-accuracy.svm(data.feature2,10);SVM2
SVM3<-accuracy.svm(data.feature3,10);SVM3

Fold<-c(1:10);Fold[11]<-'avg';Fold[12]<-'std'


cbind(Fold,naivebayes1,naivebayes2,naivebayes3,SVM1,SVM2,SVM3)

recall.nb<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  no.col<-ncol(data)
  names.class<-levels(data[,no.col-1])
  no.class<-length(names.class)
  recall<-matrix(0,nrow=k,ncol=no.class)
  colnames(recall)<-names.class
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    nb1<-naiveBayes(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.nb1<-predict(nb1,testset[,1:(ncol(data)-2)])
    table.nb<-table(pre.nb1,testset[,(ncol(data)-1)])
    rec<-diag(table.nb)/colSums(table.nb)
    recall[i,]<-rec
  }
  return(colMeans(recall))
}
reca.nb2<-recall.nb(data.feature2,10)

precision.nb<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  no.col<-ncol(data)
  names.class<-levels(data[,no.col-1])
  no.class<-length(names.class)
  precision<-matrix(0,nrow=k,ncol=no.class)
  colnames(precision)<-names.class
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    nb1<-naiveBayes(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.nb1<-predict(nb1,testset[,1:(ncol(data)-2)])
    table.nb<-table(pre.nb1,testset[,(ncol(data)-1)])
    pre<-diag(table.nb)/rowSums(table.nb)
    precision[i,]<-pre
  }

  return(colMeans(precision,na.rm=T))
}
pree.nb2<-precision.nb(data.feature2,10)

precision.nb<-pree.nb2*100
recall.nb<-reca.nb2*100
fmeasure.nb<-2*precision.nb*recall.nb/(precision.nb+recall.nb)
result.nb<-cbind(recall.nb,precision.nb,fmeasure.nb);result.nb 
macro.avg.nb<-colMeans(result.nb);macro.avg.nb
micro.avg.nb<-naivebayes2[11];micro.avg.nb 


svm
recall.svm<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  no.col<-ncol(data)
  names.class<-levels(data[,no.col-1])
  no.class<-length(names.class)
  recall<-matrix(0,nrow=k,ncol=no.class)
  colnames(recall)<-names.class
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    svm1<-svm(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.svm1<-predict(svm1,testset[,1:(ncol(data)-2)])
    table.svm<-table(pre.svm1,testset[,(ncol(data)-1)])
    rec<-diag(table.svm)/colSums(table.svm)
    recall[i,]<-rec
  }
  return(colMeans(recall))
}
reca.svm1<-recall.svm(data.feature1,10)

precision.svm<-function(data,k){
  set.seed(7755)
  data$id<-sample(1:k,nrow(data),replace=TRUE)
  list<-1:k
  no.col<-ncol(data)
  names.class<-levels(data[,no.col-1])
  no.class<-length(names.class)
  precision<-matrix(0,nrow=k,ncol=no.class)
  colnames(precision)<-names.class
  for (i in 1:k){
    trainingset<-subset(data,id %in% list[-i])
    testset<-subset(data,id %in% list[i])
    svm1<-svm(trainingset[,1:(ncol(data)-2)], trainingset[,(ncol(data)-1)])
    pre.svm1<-predict(svm1,testset[,1:(ncol(data)-2)])
    table.svm<-table(pre.svm1,testset[,(ncol(data)-1)])
    pre<-diag(table.svm)/rowSums(table.svm)
    precision[i,]<-pre
  }
 
  return(colMeans(precision,na.rm=T))
}
pree.svm1<-precision.svm(data.feature1,10)

precision.svm<-pree.svm1*100
recall.svm<-reca.svm1*100
fmeasure.svm<-2*precision.svm*recall.svm/(precision.svm+recall.svm)
result.svm<-cbind(recall.svm,precision.svm,fmeasure.svm);result.svm 
macro.avg.svm<-colMeans(result.svm);macro.avg.svm
micro.avg.svm<-SVM1[11];micro.avg.svm #micro precision=recall=f measure= accuracy




Clustering
---------------

Hierarchical Clustering 
----
data.scale<-scale(data.feature1[,-179])
d<-dist(data.scale,method='euclidean')
fit<-hclust(d,method='ward.D')
plot(fit,main='Hierarchical Clustering') 
hc<-cutree(fit,k=2) 
rect.hclust(fit,k=2) 

k means
----
km<-kmeans(data.scale,2)
c<-km$cluster;c
install.packages('fpc')
library('fpc')
plotcluster(data.scale,km$cluster)
title( main="k-means clustering Reuters data"
       ,sub=paste("R", format(Sys.time(), "%Y-%b-%d %H:%M:%S")
                  , Sys.info()["user"])) 

pam
-----
install.packages('cluster')
library('cluster')
PAM<-pam(data.scale,2)
plotcluster(data.scale,PAM$cluster)
title( main="PAM clustering Reuters data")

measure the cluster quality, use silhouette width
----
si.hc<-silhouette(hc,d)
si.km<-silhouette(km$cluster,d)
si.pam<-silhouette(PAM)
summary(si.km);summary(si.hc);summary(si.pam)#mean si wid: 0.3962-km; 0.4249-hc;  0.08338-pam
plot(si.pam)
