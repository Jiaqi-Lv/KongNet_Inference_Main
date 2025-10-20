# python3 inference_MIDOG.py \
#     --input_dir /home/u1910100/Github/KongNet_Inference_Main/test_input \
#     --output_dir /home/u1910100/Github/KongNet_Inference_Main/test_output \

python3 inference_MIDOG.py \
    --input_dir /media/u1910100/data/slides \
    --output_dir /media/u1910100/data/overlays \
    --cache_dir /home/u1910100/Documents/cache \
    --weights_dir /media/u1910100/data/GitHub/KongNet_Inference_Main/model_weights \
    --num_workers 6 \
    --batch_size 32

# python3.12 inference_CoNIC.py \
#     --input_dir /dcs/pg23/u1910100/storage/Github/KongNet_Inference_Main/test_input \
#     --output_dir /dcs/pg23/u1910100/storage/Github/KongNet_Inference_Main/test_output \
#     --cache_dir /dcs/pg23/u1910100/Documents/cache \
#     --weights_dir /dcs/pg23/u1910100/storage/Github/KongNet_Inference_Main/model_weights \
#     --num_workers 16 \
#     --batch_size 64

