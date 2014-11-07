#! /bin/sh -f

/bin/rm -fr model
mkdir model

Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./mgctm_mt
  --cor_train=/data0/data/comment/mobilePhone/LDATrainData
  --cor_test=/data0/data/comment/mobilePhone/LDATestData
  --local_topic_num=10
  --group=10
  --global_topic_num=10
  --gamma=1
  --local_alpha=0.01
  --global_alpha=0.01
  --em_iterate=100
  --var_iterate=5
  "

gdb="
  gdb ./mgctm_mt
  "
#exec $gdb
exec $cmd
