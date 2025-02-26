#!/bin/bash

#SBATCH --job-name=evaluate_compressor_NYX
#SBATCH --account=$XXX
#SBATCH --partition=$XXX
#SBATCH --nodes=1
#SBATCH --output=evaluate_compressor_NYX.txt
#SBATCH --error=evaluate_compressor_NYX.error
#SBATCH --mem-per-cpu=100gb
#SBATCH --time=02:00:00

# Setup My Environment
export OMP_NUM_THREADS=36



DIR_1=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/baryon_density.f32
DIR_2=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/dark_matter_density.f32
DIR_3=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/temperature.f32
DIR_4=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/velocity_x.f32
DIR_5=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/velocity_y.f32
DIR_6=$XXX/SDRBENCH-EXASKY-NYX-512x512x512/velocity_z.f32

# Run My Program
cd $XXX/ZCCL/install/bin
echo -e "\n"
echo -e "Evaluate hZ-dynamic with traditional DOC workflow\n"

# Initial value and its exponent
value=1E-1
exponent=-1

# Loop over the values
while [ $exponent -ge -4 ]; do
    echo -e "\n"
    echo -e "Error bound is "$value"\n"
    # Loop over the directory variables
    for i in {1..6}; do
        # Dynamically generate the variable name
        varname="DIR_$i"
        
        # Dereference the variable name to get its value
        filename="${!varname}"
        echo -e "Dataset: $filename\n"
        # Run the command with the current value and filename
        ./testfloat_homocompress_fastmode1_arg "$filename" "$filename" 36 "$value"
    done
    
    # Prepare for the next iteration of the outer loop: decrease exponent by 1
    # and update the value accordingly
    ((exponent--))
    value="1E$exponent"
done