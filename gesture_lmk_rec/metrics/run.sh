basepath=$(cd $(dirname $0); pwd)


cd $basepath/../../
source env.sh
cd $basepath/../
source setting.conf
cd $basepath

python eval.py --groundtruth_file_dir=$GROUNDTRUTH_FILE_DIR --predict_file_dir=$PREDICT_RESULT_FILE_DIR --result_json_file=$RESULT_JSON_FILE

