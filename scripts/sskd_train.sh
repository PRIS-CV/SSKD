#!/bin/sh
SOURCE=dukemtmc 
TARGET=market1501         # market1501  dukemtmc
ARCH=resnet50
CLUSTER=500


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python examples/sskd_train.py -ds ${SOURCE} -dt ${TARGET} -a ${ARCH} \
	--num-instances 4 --lr 0.00035 --iters 400 -b 64 --epochs 20 \
	--weight 0.5 --weight_ms 3 --weight_tf 1.5 --dropout 0 --lambda-value 0 \
	--init-1 /data/yinjunhui/data/per-id/logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-1/model_best.pth.tar \
	--init-2 /data/yinjunhui/data/per-id/logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-2/model_best.pth.tar \
	--init-3 /data/yinjunhui/data/per-id/logs/${SOURCE}TO${TARGET}/${ARCH}-pretrain-3/model_best.pth.tar \
	--logs-dir /data/yinjunhui/data/per-id/logs/${SOURCE}TO${TARGET}/${ARCH}-MMT-DBSCAN
	# --rr-gpu
