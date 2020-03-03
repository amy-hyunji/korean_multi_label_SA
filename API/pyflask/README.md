# API for adapter-bert sentiment analysis model

## How to install CHROMEDRIVER
1. find the version of your chrome
2. download the corresponding version of chromedriver from link below
http://chromedriver.storage.googleapis.com/index.html
place the chromedriver in **driver** folder

## REVISION NEEDED
- CKPT_DIR : specific directory of the model checkpoint 
			  : bert-adapter-tfhub_models_korean_sa_model.ckpt-3104
- OUTPUT_DIR : directory that contains all the checkpoints
- CONFIG_DIR : directory that contains bert_config.json
- path in result function : should have chromedriver to use crawling

place checkpoints of 99_checkpoint.tar.gz under **4_checkpoint** and checkpoint.tar.gz in **checkpoints**

## COMMAND
1. for version1 5 emotions
<pre><code>python3 app.py</code></pre>
2. for update version with 4 emotions
<pre><code>python3 4_app.py</code></pre>

## HOW TO ACCESS
localhost:5000

