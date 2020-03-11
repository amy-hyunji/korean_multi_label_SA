# BERT + adapter for Korean SA finetuning
multi labeled korean sentiment analysis
- 5 labels: neutral, happy, sad, angry, surprised
- multilingual BERT model
- Google adapter for better transfer learning


## checkpoint URL
https://drive.google.com/open?id=12HOjadTU04waQ_yuKOQWNZU0lcWPQY-j


## How to run
1. Upload train/Bert_adapter.ipynb on colab
2. In google cloud platform, create a storage named 'mbertfinetune'
3. Create a subdirectory data/korean_sa
4. Upload dataset on the subdirectory

- on 3rd cell, to classify 5 emotions: <pre><code>TASK = 'korean_sa'</code></pre>, to classify 4 emotions(without neutral) : <pre><code>TASK = 'korean_sa_4'</code></pre>
- on 4th cell, adjust hyperparameters such as lr and epochs
- on 4th cell, layer_wise_lr applies different learning rates on different layers
    (ex) layer_wise_lr = (True, 0.3) => init_lr on top layer, init_lr * 0.3 on second to top layer ..

- output will be saved to your storage (gs://mbertfinetune/bert-adapter-tfhub/models/korean_sa_4 or korean_sa)


## Adapter-Bert model
slight changes were made to run this specific task
- original adapter-Bert : https://github.com/google-research/adapter-bert
- adapter-Bert for this task: https://github.com/junhahyung/adapter-bert


## BOG_final.ipynb
classify labels by Bag_Of_Words and linear SVM
- "report_results(grid_svm.best_estimator_, X_test, y_test" cell will return 
    1. accuracy
    2. F1 score
    3. precision
    4. recall
- takes about 2hrs for 20000 sentences


## Result
### BOG_final.ipynb
- score with "without_neutral_train/test"
- takes about 4.5 hrs
1. Accuracy : 0.6057255676209279
2. F1 score : [0.65456652, 0.61592795, 0.55240913, 0.59325651]
3. Precision :  [0.65242494, 0.56458401, 0.6188447 , 0.5963106 ]
4. Recall : [0.6567222 , 0.67754468, 0.49885496, 0.59023355]
### Adapter-Bert
1. VERSION1 with 5 emotions 
- EPOCH 3\
**[F1 score]**\
neutral_f1: 0.60294\
anger_f1: 0.76289\
happy_f1: 0.66069\
sad_f1:  0.64336\
surprised_f1: 0.60115\
**[confusion matrix]**\
neutral     [410 27 82 67 113]\
happy       [38 518 57 52 35]\
sad         [90 28 479 60 43]\
angry       [45 39 69 460 84]\
surprise    [78 46 63 94 419]

2. VERSION2 with 4 emotions (without NEUTRAL)
- EPOCH 100\
**[F1 score]**\
anger_f1: 0.6947900775643995\
happy_f1: 0.7604186858082728\
sad_f1: 0.7242876108892842\
surprised_f1: 0.666947331666022\
**[confusion matrix]**\
happy       [1925, 221, 196, 238]\
sad         [178, 1919, 243, 234]\
angry       [195, 298, 1787, 339]\
surprise    [185, 287, 299, 1584]


TODO
adapter없는 애랑도 비교해서 돌려보기
