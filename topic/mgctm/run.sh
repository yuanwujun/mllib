#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./mgctm
  --cor_path=/data0/data/order3/order3_1w_sku_user_lda_data
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
