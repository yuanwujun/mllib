#!/bin/sh -f

/bin/rm -fr model
mkdir model

Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./var_lda_asy
  --cor_train=/data0/data/comment/mobilePhone/LDATrainData
  --cor_test=/data0/data/comment/mobilePhone/LDATestData
  --alpha=0.01 
  --em_iterate=30
  --var_iterate=30
  --topic_num=10
  "

gdb="
  gdb ./var_lda_asy
  "
#exec $gdb
exec $cmd
