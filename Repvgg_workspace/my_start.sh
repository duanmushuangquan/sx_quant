# 启动my_ptq的量化命令
# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 01Origin.py --arch RepVGG-A0 --data-path /home/lr/workspace/data/imagenet --batch-size 128 --tag test --eval --resume ./RepVGG_Pretrained/RepVGG-A0-train.pth --opts DATA.DATASET imagenet DATA.IMG_SIZE 224

# 启动aaaaa.py的量化命令
# --batch-size 128          训练集、验证集dataloader的batch_size
# --calib_batch_size 50     标定多少恶bathch_size
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12349 02Simplified_quant.py --arch RepVGG-A0 --data-path /home/lr/workspace/data/imagenet --batch-size 128 --calib_batch_size 50  --tag test --eval --resume ./RepVGG_Pretrained/RepVGG-A0-train.pth --opts DATA.DATASET imagenet DATA.IMG_SIZE 224