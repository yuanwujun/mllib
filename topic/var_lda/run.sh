#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./var_lda
  --cor_path=./data/ap.dat
  --alpha=0.01
  --topic_num=10
  "

gdb="
  gdb ./rtm
  "
#exec $gdb
exec $cmd
