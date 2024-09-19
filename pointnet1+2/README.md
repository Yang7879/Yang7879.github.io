# pytorch_Comment_Pointnet1-2
dataset 和train.py在同一目录下
shapenetdataset:downloadfile.sh

SCRIPT=`realpath $0`
SCRIPTPATH=`dirname $SCRIPT`

cd $SCRIPTPATH/..
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0.zip
rm shapenetcore_partanno_segmentation_benchmark_v0.zip
cd -

modelnet40dataset:
https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
然后将储存目录的两个文件modelnet40_test.txt和modelnet40_train.txt的内容更改为如下格式
toilet_0412 -> toilet/toilet_0412
