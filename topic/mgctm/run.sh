#! /bin/sh -f

/bin/rm -fr model
mkdir model

Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./mgctm
  --cor_train=/data0/data/comment/655/lda_data_3000
  --cor_test=/data0/data/comment/655/lda_data_3000
  --local_topic_num=10
  --group=10
  --global_topic_num=10
  --gamma=1
  --local_alpha=0.01
  --global_alpha=0.01
  "

gdb="
  gdb ./mgctm
  "
#exec $gdb
exec $cmd
