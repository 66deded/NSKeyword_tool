# NSKeyword Tool

## Project Description

The NSKeyword Tool is a detection tool for NOKESCAM websites in search engines, based on ***jieba*** segmentation and the ***BERT*** model. 
It supports word segmentation of Chinese titles, filtering of meaningless words, and classification of malicious titles based on semantic analysis.


## Install Dependencies

Before running the project, ensure the following Python libraries are installed:

```bash
pip install numpy torch tqdm jieba transformers
```

## Usage Instructions

1. **Prepare Data**:

   - Place the texts to be processed in `NSKeyword_tool/datas/data/input.txt`, one per line.

2. **Segmentation**:

   - Run the following command for analysis:
   ```
   python demo.py --model bert
   ```

3. **Meaningless Word Filtering**:

   - After segmentation, the tool will automatically filter out meaningless words based on rules, saving results to `filtered_output.txt`.

4. **Model Prediction**:
   - The tool classifies the filtered titles, saving the final results to `res.txt`.
   - Classification codes `[0, 1, 2, 3]` correspond to `[Normal, Fake Online Game, Pornographic, Gambling]`.


## Main Features

- **Chinese Segmentation**: Utilizes the `jieba` library for text segmentation.
- **Title Filtering**: Filters out irrelevant texts based on predefined keywords.
- *Title Classification**: Classifies texts using a trained `BERT` model.


## File Description

- `bert-base-chinese/`: Contains BERT model configuration and weight files.
- `data/`: Stores input data and model checkpoint files.
- `models/`: Includes implementation code and utility functions for the model.
- `filtered_output.txt`: Stores filtered text results.
- `res.txt`: Contains final prediction results.
- `segmented_output.txt`: Stores segmented text results.

