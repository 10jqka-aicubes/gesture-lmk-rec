import os, json
import numpy as np
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file_dir", type=str, default="../result/")
    parser.add_argument("--groundtruth_file_dir", type=str, default="../groundtruth/")
    parser.add_argument("--result_json_file", type=str, default="../result.json")
    args = parser.parse_args()
    try:
        f1 = open(os.path.join(args.predict_file_dir, "predict.txt")).readlines()
        f2 = open(os.path.join(args.groundtruth_file_dir, "groundtruth/data_label.txt")).readlines()
        num = len(f2)
        count_lmk = 0
        count_gesture = 0
        count = 0
        score = 0
        for i in range(num):
            line1, line2 = f1[i], f2[i]
            line1 = line1.strip().split()
            line2 = line2.strip().split()
            yt = line2[1]
            yp = line1[1]
            lt = np.asarray(line2[6:], dtype=np.float32)
            lp = np.asarray(line1[2:], dtype=np.float32)
            lt = lt.reshape([21, 2])
            lp = lp.reshape([21, 2])
            nme = np.sum(np.linalg.norm(lp - lt, axis=1) / 21)
            count += 1
            if nme < 0.03:
                count_lmk += 1
            if yt == yp:
                count_gesture += 1
            score = count_lmk / count * 40 + count_gesture / count * 60
        print(score)
        result = {"score": score}
        with open(args.result_json_file, "w", encoding="utf-8") as fout:
            fout.write(json.dumps(result))
    except Exception as e:
        print(e)
