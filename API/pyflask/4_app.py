from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import os
from flask import Flask, render_template, request
from sklearn.externals import joblib
app = Flask(__name__)

# for BERT
import datetime
import json
import random
import string
import sys
import csv
import pandas as pd
import tensorflow as tf

# import python modules define by BERT
import modeling
import optimization
import run_classifier
import run_classifier_with_tfhub
import tokenization
import tensorflow_hub as hub

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 1
LEARNING_RATE = 3e-4
NUM_TRAIN_EPOCHS = 3
MAX_SEQ_LENGTH = 128
WARMUP_PROPORTION = 0.1
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 300

TASK = "korean_sa"
BUCKET = "mbertfinetune"
BERT_MODEL = 'multi_cased_L-12_H-768_A-12'
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_'+BERT_MODEL+'/1'

CKPT_DIR = "../../../99_checkpoint/multi_single_naver/bert-adapter-tfhub_models_korean_sa_4_model.ckpt-219918"
OUTPUT_DIR= "../../../99_checkpoint/multi_single_naver"
CONFIG_DIR = "../../../99_checkpoint/bert/bert_config.json"

@app.route("/")
def index():
    return render_template("main.html") 

@app.route("/result",methods=['POST','GET'])
def result():
    if request.method=="POST":
        result = request.form.to_dict()
        
        search_sentence = result['Text']
        path = "../../../chromedriver"
        chrome_options = webdriver.ChromeOptions()
        chrome_options.add_argument('headless')	

        driver = webdriver.Chrome(path, chrome_options=chrome_options)
        driver.get("http://speller.cs.pusan.ac.kr/")
        search_box = driver.find_element_by_name("text1")
        search_box.send_keys(search_sentence)
        search_box.submit()
        
        m=0
        while(1):
            try:
                _new = driver.find_element_by_id("ul_{}".format(m))
                _new_array = _new.text.split("\n")
                revised = _new_array[0]
                original = _new_array[1]
                search_sentence = search_sentence.replace(original, revised, 1)
                m = m+1
            except:
                break

        result["Text"] = search_sentence

        # CLASSIFY
        emotion = classify(search_sentence)
        result.update(emotion)
        results = [result]
        
        return render_template("4_emotion.html", results=results)

def get_run_config():
    return tf.contrib.tpu.RunConfig(
        cluster = None,
        model_dir = OUTPUT_DIR,
        save_checkpoints_steps = SAVE_CHECKPOINTS_STEPS,
        tpu_config = tf.contrib.tpu.TPUConfig(
            iterations_per_loop = 1000,
            num_shards = 8,
            per_host_input_for_training = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        )
    )

def classify(search_sentence):

    examples = []
    guid = "dev-1"
    text_a = tokenization.convert_to_unicode(search_sentence)
    label = tokenization.convert_to_unicode("1") # put dummy input
    examples.append(run_classifier.InputExample(guid=guid, text_a=text_a, text_b=None, label=label))

    predict_features = run_classifier.convert_examples_to_features(examples, label_list, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = run_classifier.input_fn_builder(
        features = predict_features,
        seq_length = MAX_SEQ_LENGTH,
        is_training = False,
        drop_remainder = True)
    result = estimator.predict(input_fn = predict_input_fn)
    for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i>= 1: break
        
        emotion = {"happy": 0, "sad": 0, "angry": 0, "surprised": 0}
        _emotion = ["happy", "sad", "angry", "surprised"]
        for (i, class_probability) in enumerate(probabilities): 
           emotion[_emotion[i]] = class_probability	
           print(_emotion[i] + ": " + str(class_probability))

    return emotion

def initsetting():
    print("DOING INITIALSETTING!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    tokenizer = run_classifier_with_tfhub.create_tokenizer_from_hub_module(BERT_MODEL_HUB)

    processors = {
        "cola": run_classifier.ColaProcessor,
        "mnli": run_classifier.MnliProcessor,
        "mrpc": run_classifier.MrpcProcessor,
		  "korean_sa": run_classifier.KsaProcessor_4,
    }	
    processor = processors[TASK.lower()]()
    label_list = processor.get_labels()

    num_train_steps = 1
    num_warmup_steps = None

    bert_config = modeling.BertConfig.from_json_file(CONFIG_DIR)
    model_fn = run_classifier.model_fn_builder(
        bert_config = bert_config,
        num_labels = len(label_list),
        init_checkpoint = CKPT_DIR,
        learning_rate = 3e-4,
        num_train_steps = num_train_steps,
        num_warmup_steps = num_warmup_steps,
        use_tpu = False,
        use_one_hot_embeddings = False
        )

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu = False,
        model_fn = model_fn,
        config = get_run_config(),
        train_batch_size = 32,
        eval_batch_size = 8,
        predict_batch_size = 1
        )
    return tokenizer, estimator, label_list  

if __name__=="__main__":
    tokenizer, estimator, label_list = initsetting()
    app.run(host="127.0.0.1", port=5000, debug=True)
