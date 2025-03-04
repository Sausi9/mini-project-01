# Run the training script 10 times
for i in {1..10}
do
    {
        echo "=== Training Run $i ==="
        python3 src/mini_project_01/train.py
    } >> all_output_vamp.txt
done
