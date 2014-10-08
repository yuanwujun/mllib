#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./rtm
  --net_path=./data/network
  --cor_path=./data/rtm_corpus
  --alpha=0.01
  --topic_num=10
  "

gdb="
  gdb ./rtm
  "
#exec $gdb
exec $cmd
