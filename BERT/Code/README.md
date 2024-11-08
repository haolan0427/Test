1. [`下载预训练的模型：`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

   解压缩，将其中的文件移动到./PretrainedModel目录下

   执行以下命令以运行convert.py

   ```bash
   # pycharm-Windows
   .\.venv\Scripts\python.exe convert.py --tf_checkpoint_path ./PretrainedModel/bert_model.ckpt --bert_config_file ./PretrainedModel/bert_config.json --pytorch_dump_path ./PretrainedModel/pytorch_model.bin
   ```

2. 下载训练集：通过百度网盘分享的文件：train.tsv
   链接：https://pan.baidu.com/s/1F1wU4-JK0vpZz9GyVSNDxg?pwd=5nft 
   提取码：5nft和下载测试集：通过百度网盘分享的文件：dev.tsv
   链接：https://pan.baidu.com/s/1A6hWaRZNntPPOecoiavzOQ?pwd=s22f 
   提取码：s22f。下载完毕后，放置在./data目录下

3. 执行以下命令以运行run_classifier_word.py

   ```python
   # pycharm-Windows
   .\.venv\Scripts\python.exe run_classifier_word.py --task_name NEWS --do_train --do_eval --data_dir ./data --vocab_file ./PretrainedModel/vocab.txt --bert_config_file ./PretrainedModel/bert_config.json --init_checkpoint ./PretrainedModel/pytorch_model.bin --max_seq_length 256 --train_batch_size 24 --learning_rate 2e-5 --num_train_epochs 50.0 --output_dir ./newsAll_output
   ```


4. 调试，在debug.py打断点

   
