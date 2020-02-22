# BERT + adapter for Korean SA finetuning
multi labeled korean sentiment analysis
- 5 labels: neutral, happy, sad, angry, surprised
- multilingual BERT model
- Google adapter for better transfer learning


## checkpoint URL
https://drive.google.com/file/d/1UkYxxvygFfuj3pjhHcm3glwdp3rPn49V/view?usp=sharing

## How to run
1. Upload train/Bert_adapter.ipynb on colab
2. In google cloud platform, create a storage named 'mbertfinetune'
3. Create a subdirectory data/korean_sa
4. Upload dataset on the subdirectory

- on 3rd cell, set <pre><code>TASK = 'korean_sa'</code></pre> to classify 5 emotions, set <pre><code>TASK = 'korean_sa_4'</code></pre> to classify 4 emotions(without neutral)
- on 4th cell, adjust hyperparameters such as lr and epochs
- on 4th cell, layer_wise_lr applies different learning rates on different layers
    (ex) layer_wise_lr = (True, 0.3) => init_lr on top layer, init_lr * 0.3 on second to top layer ..

- output will be saved to your storage (gs://mbertfinetune/bert-adapter-tfhub/models/korean_sa_4 or korean_sa)

## adapter-Bert model
slight changes were made to run this specific task
- original adapter-Bert : https://github.com/google-research/adapter-bert
- adapter-Bert for this task: https://github.com/junhahyung/adapter-bert

## result


