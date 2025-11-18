#!/bin/bash
# This script runs the demos to test pull request #3

echo " "
echo "Running file test_pr1.sh: Test for pull request #1"
echo " "

# Installs package
echo " "
echo "Installing wind_density_tomo package"
echo " "
cd ../dev_scripts
source clean_install_all.sh

# Downloads data for the depot
echo " "
echo "Downloading data from the depot"
echo " "
cd ../demo
source get_demo_data_server.sh

# Runs demo file
cd ..
echo " "
echo "Running demo file: demo_simulation_and_tomography.py"
echo " "
python demo/demo_simulation_and_tomography.py

echo " "
echo "Test for pull request #1 complete"
echo " "