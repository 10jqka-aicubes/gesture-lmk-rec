basepath=$(cd $(dirname $0); pwd)
###
 # @Author: byron
 # @Descripttion: 
 # @version: 
 # @Date: 2022-07-27 09:34:53
 # @LastEditors: way
 # @LastEditTime: 2022-07-27 09:34:54
### 


cd $basepath/../../
source env.sh
cd $basepath/../

source setting.conf
cd $basepath
python train.py \
     --train_data_dir=$TRAIN_FILE_DIR \
     --model_dir=$SAVE_MODEL_DIR 
