#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./mgctm
  --cor_path=/data0/data/order3/order3_8000_lda_data
  --alpha=0.01
  --topic_num=10
  "

gdb="
  gdb ./mgctm
  "
#exec $gdb
exec $cmd
