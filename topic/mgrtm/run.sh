#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./mgrtm
  --net_path=./data/rtm_network
  --cor_path=./data/rtm_corpus
  --alpha=0.01
  --local_topic=10
  --global_topic=10
  --neg_times=10
  "

#/data0/data/rtm/link    ./data/network
#/data0/data/rtm/corpus   ./data/rtm_corpus

gdb="
  gdb ./mgrtm
  "
#exec $gdb
exec $cmd
