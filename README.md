# 【7-8暑期赛】自由场景下的大角度手势识别算法

&emsp;&emsp;自由场景不限角度的手掌关键点检测及手势识别是计算机视觉中的研究难点，近些年随着技术的进步，手掌关键点辅助于手势识别在零售，教育等行业运用越发普及，但是手掌关键点在实际落地过程中往往存在以下挑战： 1：受背景，光线，遮挡等因素干扰，误判率较高。 2：在一般运动过程中，手掌因角度，姿态等因素，会导致手掌各个关键点相互遮挡，影响识别精度。本项目旨在解决静态手势中常见的关键点检测与手势识别，数据集提供了剪刀石头布交互场景下的静态图片。

- 本代码是该赛题的一个基础demo，仅供参考学习
- 比赛地址：<http://contest.aicubes.cn/>
- 时间：2022-07 ~ 2022-08

## 如何运行demo
- clone代码
- 准备环境 
  - cuda11.0以上
  - python3.7以上
  - 安装python依赖 
  - python -m pip install -r requirements.txt
- 准备数据，从官网[下载数据](http://contest.aicubes.cn/#/detail?topicId=83)
- 调整参数配置，参考[模板项目](https://github.com/10jqka-aicubes/project-demo)的说明，主要修改的配置
  - gesture_lmk_rec/setting.conf
- 运行
   - 训练
   ```
   bash gesture_lmk_rec/train/run.sh
   ```
   - 预测
   ```
   bash gesture_lmk_rec/predict/run.sh
  ```
   - 计算结果指标
   ```
   bash gesture_lmk_rec/metrics/run.sh
   ```
   **为防止实现计算评测的公式有差异，可参考使用对应赛题提供的demo中的代码**


## 反作弊声明
1）参与者不允许使用多个小号，一经发现将取消成绩；

2）参与者禁止在指定考核技术能力的范围外利用规则漏洞或技术漏洞等途径提高成绩排名，经发现将取消成绩；

3）在A榜中，若主办方认为排行榜成绩异常，需要参赛队伍配合给出可复现的代码

## 赛事交流
![同花顺比赛小助手](http://speech.10jqka.com.cn/arthmetic_operation/245984a4c8b34111a79a5151d5cd6024/客服微信.JPEG)