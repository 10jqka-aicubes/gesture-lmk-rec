basepath=$(cd $(dirname $0); pwd)
###
 # @Author: byron
 # @Descripttion: 
 # @version: 
 # @Date: 2022-07-26 20:52:35
 # @LastEditors: way
 # @LastEditTime: 2022-07-27 10:45:27
### 

cd $basepath/../../
source env.sh
cd $basepath/../

source setting.conf
cd $basepath
python predict.py \
     --data_dir=$PREDICT_FILE_DIR --model_dir=$SAVE_MODEL_DIR --result_dir=$PREDICT_RESULT_FILE_DIR
     
