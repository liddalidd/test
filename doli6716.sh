spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --num-executors 3 \
        doli6716A2.py \
        --output $1
