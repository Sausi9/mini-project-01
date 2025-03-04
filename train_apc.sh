# Run the training script 10 times on Arnór Performance Cluster (APC) for vae experiments.
# Rename the output file to the appropriate prior name.
for i in {1..10}
do
    {
        echo "=== Training Run $i ==="
        python src/mini_project_01/train.py
    } >> all_output_gaussian.txt
done