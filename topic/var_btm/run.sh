#!/bin/sh -f

/bin/rm -fr model
mkdir model

Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./var_btm
  --cor_train=/data0/data/btm/biterm
  --cor_test=/data0/data/btm/biterm
  --beta=0.01 
  --em_iterate=30
  --var_iterate=30
  --topic_num=10
  "

gdb="
  gdb ./var_btm
  "
exec $gdb
#exec $cmd
