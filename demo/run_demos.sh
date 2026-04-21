#!/bin/bash
# This script runs the demos

echo " "
echo "Running file run_demos.sh"
echo " "

# Install package only if not already installed
if python -c "import winddensity_mbir" 2>/dev/null; then
    echo "winddensity_mbir already installed, skipping."
else
    echo "Installing winddensity_mbir package"
    cd ../dev_scripts
    source clean_install_all.sh
fi

# Runs demo files
cd ..
echo " "
echo "Running demo file: demo_simulation_and_tomography.py"
echo " "
python demo/demo_simulation_and_tomography.py

echo " "
echo "Running demo file: demo_processing_experimental_data.py"
echo " "
python demo/demo_processing_experimental_data.py


echo " "
echo "All demos completed"
echo " "