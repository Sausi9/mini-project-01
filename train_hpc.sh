#!/bin/bash
### ADJUST THIS BASED ON WHAT GPU TO USE
#BSUB -q gpua100


### Set the job name
#BSUB -J mini_project_01

### Request the number of CPU cores
#BSUB -n 6
#BSUB -R "span[hosts=1]"
### Select GPU resources
#BSUB -gpu "num=1:mode=exclusive_process"

### Set walltime limit: hh:mm
# Maximum 24 hours for GPU queues
#BSUB -W 01:30

### Request memory per core
# Adjust based on your application's requirements
# Example: 10GB per core
#BSUB -R "rusage[mem=80GB]"


### Specify output and error files
#BSUB -o train_%J.out
#BSUB -e train_%J.err

# =============================================================================
# Environment Setup
# =============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# =============================================================================
# Module Loading Based on Selected Queue
# =============================================================================

# Ensure that only one queue is uncommented
selected_queues=($(echo "$LSB_QUEUE"))

# Since LSF directives are processed before the script runs, we need another way
# to ensure only one queue is selected. We'll count the number of uncommented queues.

queue_count=$(grep -c '^#BSUB -q ' train_hpc.sh)

if [ "$queue_count" -ne 1 ]; then
    echo "Error: Please uncomment exactly one queue in the job script."
    exit 1
fi

# Load modules based on the selected queue

module load cuda/12.6.2
module load intel/2024.2.mpi
module load mpi/5.0.5-gcc-14.2.0-binutils-2.43 
module load python3/3.11.10

# =============================================================================
# Environment Setup
# =============================================================================

# Define variables
REQUIREMENTS_TXT="requirements.txt"
VENV_DIR=".venv"

if [ ! -d $VENV_DIR ]; then
    python3 -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    echo "====RUNNING PYTHON FROM===="
    which python
    echo "====INSTALLING REQUIREMENTS===="
    pip install -r $REQUIREMENTS_TXT
    echo "====REQUIREMENTS INSTALLED===="
else 
    echo "====ACTIVATING EXISTING ENVIRONMENT===="
    source $VENV_DIR/bin/activate
fi
# =============================================================================
# Training Execution
# =============================================================================

echo "=== Starting Training ==="

# Navigate to the directory containing train.py if not already there
# Uncomment and modify the following line if necessary
# cd /path/to/your/project

# Execute the training script
# python3 src/mini_project_01/train.py

# Run script 10 times
for i in {1..3}
do
    echo "=== Training Run $i ==="
    python -u src/mini_project_01/train.py
done    



echo "=== Training Completed ==="

# =============================================================================
# Final Steps
# =============================================================================

# Optionally, deactivate the Conda environment
deactivate

echo "=== HPC Job Script Completed ==="