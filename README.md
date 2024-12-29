# NSKeyword Tool

## 项目简介

NSKeyword Tool 是一个用于判断搜索引擎中的 NOKESCAM 网站的检测工具，基于 ***jieba** 分词和 **BERT*** 模型实现。该工具支持对网站中文标题的分词、无意义词判断过滤及基于标题语义的恶意标题分类。

## 目录结构

NSKeyword_tool/
 │
 ├── bert-base-chinese/
 │   ├── config.json
 │   ├── pytorch_model.bin
 │   ├── README.md
 │   └── vocab.txt
 │
 ├── data/
 │   ├── class.txt
 │   ├── input.txt
 │   ├── saved_dict/
 │   ├── bert.ckpt
 │   └── bert_ab.ckpt
 │
 ├── models/
 │   ├── bert.py
 │   └── utils.py
 │
 ├── filtered_output.txt
 ├── res.txt
 ├── segmented_output.txt
 └── demo.py

## 安装依赖

在运行项目之前，请确保已安装以下 Python 库：

```bash
pip install numpy torch tqdm jieba transformers
```

## 使用方法

1. **准备数据**：

   - 将待处理的文本放入 `NSKeyword_tool/datas/data/input.txt` 文件中，每行一个文本。

2. **分词处理**：

   - 运行以下命令进行检测分析：

     python demo.py --model bert

3. **无意义词过滤**：
   - 经过分词后，工具会根据规则自动筛选出对应的无意义词，并将结果保存至 `filtered_output.txt`。

4. **模型预测**：
   - 工具会对筛选出的标题集合进行分类预测，最终结果将保存到 `res.txt` 文件中。
   - 分类编号 [0, 1, 2, 3] 对应 [正常, 虚假网游, 色情, 博彩]

## 主要功能

- **中文分词**：使用 `jieba` 库进行中文文本的分词处理。
- **标题过滤**：根据预定义的关键词过滤掉不相关的文本。
- **标题分类**：使用 训练后的 BERT 模型对文本进行分类预测。

## 文件说明

- `bert-base-chinese/`：包含 BERT 模型的配置和权重文件。
- `data/`：存放输入数据和模型检查点文件。
- `models/`：包含模型的实现代码和辅助工具函数。
- `filtered_output.txt`：存储经过过滤的文本结果。
- `res.txt`：存储最终预测结果。
- `segmented_output.txt`：存储分词后的文本结果。

