#!/bin/bash

function addhost {
  host=$1
  file=$2
  ping -c 1 -w 1 $host  &> /dev/null;
  if [ $? -eq 0 ]; then
     echo "$host slots=1"  >> $2
  fi
}

echo "Preparing..."
HOSTLIST=( frontend sisu01 sisu02 sisu03 sisu04 sisu05 sisu06 sisu07 sisu08 sisu09 )
if [ -e .hosts ]
then
    rm -f .hosts
fi

for i in ${HOSTLIST[@]}; do
    #add host to list
    addhost $i .hosts
    #turn off blink
    ssh $i /usr/sbin/blink1-tool --off -q
done;


n_hosts=$(wc -l .hosts|gawk '{print $1}')
echo "Launching on $n_hosts nodes"
/usr/lib64/openmpi/bin/mpirun -np $n_hosts --hostfile .hosts /export/alfthan/sisunen-demo-gol/gol.py  -n 1000 -r 10 -s random

for i in ${HOSTLIST[@]}; do
    #turn off blink
    ssh $i /usr/sbin/blink1-tool --off -q
done;



