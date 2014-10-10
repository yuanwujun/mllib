#! /bin/sh -f
Lib="$HOME/ywj/lib/lib"
export LD_LIBRARY_PATH="$Lib:LD_LIBRARY_PATH"
cmd="
  ./var_lda
  --cor_path=/data0/data/lda_user_sku_20-40_5000
  --alpha=0.01
  --topic_num=200
  "

gdb="
  gdb ./var_lda
  "
#exec $gdb
exec $cmd
