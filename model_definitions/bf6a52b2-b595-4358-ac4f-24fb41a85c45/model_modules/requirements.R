message('Installing packages')
if(!require('gbm')){install.packages('gbm')}
if(!require('devtools')){install.packages('devtools')}
if(!require('caret')){install.packages('caret')}
if(!require('tdplyr')){install.packages('tdplyr', repos=c('http://teradata-download.s3.amazonaws.com','http://cloud.r-project.org'))}
#library("devtools")
#install_git("git://github.com/jpmml/r2pmml.git")
