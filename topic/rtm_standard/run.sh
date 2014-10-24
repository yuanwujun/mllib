#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./rtm
  --cor_path=./data/rtm_corpus
  --net_path=./data/network
  --alpha=0.01
  --topic_num=10
  "

gdb="
  gdb ./rtm
  "
#exec $gdb
exec $cmd
