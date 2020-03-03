# API for adapter-bert sentiment analysis model

## How to install CHROMEDRIVER
1. find the version of your chrome
2. download the corresponding version of chromedriver from link below
http://chromedriver.storage.googleapis.com/index.html
place the chromedriver in **driver** folder

## INITIAL SETTING
- CKPT_DIR : specific directory of the model checkpoint 
			  : bert-adapter-tfhub_models_korean_sa_model.ckpt-3104
- OUTPUT_DIR : directory that contains all the checkpoints
- CONFIG_DIR : directory that contains bert_config.json
- path in result function : should have chromedriver to use crawling

https://drive.google.com/open?id=12HOjadTU04waQ_yuKOQWNZU0lcWPQY-j

download and place it under **API** folder with **driver** folder

## COMMAND
1. for version1 5 emotions
<pre><code>python3 app_5.py</code></pre>
2. for update version with 4 emotions
<pre><code>python3 app_4.py</code></pre>

## HOW TO ACCESS
localhost:5000

