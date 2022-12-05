1. To run 1 GPU example: <br />
python3 mnist_bert_training.py <br /><br />
2. To run distributed example:<br />
python3 -m torch.distributed.launch --nproc_per_node=2 bert_training.py 