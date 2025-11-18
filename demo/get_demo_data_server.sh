#!/bin/bash

GPUCLUSTER="gilbreth"
CPUCLUSTER="brown"

source1="/depot/bouman/users/kweisen/winddensity_mbir/data"

# Note that the target directories are relative to demo
target_dir="./"

red=`tput setaf 1`
reset=`tput sgr0`

if [[ "$HOSTNAME" == *"$GPUCLUSTER"* || "$HOSTNAME" == *"$CPUCLUSTER"* ]]; then
  account_name=""
  command="cp"
else
  echo "Enter your Purdue Career Account user name:"
  read cluster_user_name
  account_name="$cluster_user_name@gilbreth.rcac.purdue.edu:"
  command="scp"
fi

echo "${red}   Copying $source1 to $target_dir ${reset}"
$command -r "$account_name$source1" "$target_dir"
