import os
from flask import Flask, render_template, request
app = Flask(__name__)

# for BERT
import io
import numpy as np
from torch import nn
import torch.onnx
import onnx 
import onnxruntime
from Models import BertClassifier
from KoBERT.kobert.pytorch_kobert_adapter import get_pytorch_kobert_model_adapter
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model
from tqdm.notebook import tqdm
from KoBERT.kobert.utils import get_tokenizer
import gluonnlp as nlp
from load_dataset import *

ADAPTER = False
ONNX = True 

@app.route("/")
def index():
    return render_template("main.html") 

@app.route("/result",methods=['POST','GET'])
def result():
    if request.method=="POST":
        
        result = request.form.to_dict()
        print(result) 
        search_sentence = result['Text']
        NSMC = True 
        if '4way' in result.keys() and result['4way'] == 'on':
            NSMC = False 

        t_token_ids = []; t_valid_length = []; t_segment_ids = []
        if NSMC:
            t_token_id, t_val_len, t_segment_id = transform2([search_sentence])
        else:
            t_token_id, t_val_len, t_segment_id = transform4([search_sentence])
        t_token_ids.append(t_token_id)
        t_valid_length.append(int(t_val_len))
        t_segment_ids.append(t_segment_id)

        t_token_ids = torch.LongTensor(t_token_ids).long()
        t_valid_length = torch.LongTensor(t_valid_length)
        t_segment_ids = torch.LongTensor(t_segment_ids).long()
        
        if ONNX:
            if NSMC: 
					inputs = {
							  'token_ids': to_numpy(t_token_ids),
							  'valid_length': to_numpy(t_valid_length),
							  'segment_ids': to_numpy(t_segment_ids),
							  }
					val = 2_session.run(None, inputs)
				else:
					inputs = {
							  'token_ids': to_numpy(t_token_ids),
							  'valid_length': to_numpy(t_valid_length),
							  'segment_ids': to_numpy(t_segment_ids),
							  }
					val = 4_session.run(None, inputs)
        else:
            if NSMC:
                val = 2_model(t_token_ids, t_valid_length, t_segment_ids)
            else:
                val = 4_model(t_token_ids, t_valid_length, t_segment_ids)

        # CLASSIFY
        emotion = classify(val, NSMC)
        result.update(emotion)
        results = [result]
        
        if not NSMC:
            return render_template("4_emotion.html", results=results)
        else:
            return render_template("2_emotion.html", results=results)

def classify(result, NSMC):
    if not ONNX:
        result = result.detach().numpy() if result.requires_grad else result.numpy()
        result = result[0] 
    else:
        result = result[0][0]
    result = linearize(result)

    if NSMC:
        emotion = {'negative': 0, "positive": 0}
        _emotion = ['negative', 'positive']
        for (i, prob) in enumerate(result):
            emotion[_emotion[i]] = prob
         
    else:
        emotion = {"happy": 0, "sad": 0, "angry": 0, "surprised": 0}
        _emotion = ["happy", "sad", "angry", "surprised"]
        for (i, prob) in enumerate(result): 
           emotion[_emotion[i]] = prob 

    return emotion

def linearize(x):
    x = np.array(x)
    _min = np.min(x)
    if _min < 0: x += (-_min)
    _sum = np.sum(x)
    return x/_sum

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum()

def to_numpy(tensor):
    return tensor.cpu().numpy()

def initsetting():
    bertmodel2, vocab2 = get_pytorch_kobert_model_adapter()
    bertmodel4, vocab4 = get_pytorch_kobert_model()

    tokenizer = get_tokenizer()
    tok2 = nlp.data.BERTSPTokenizer(tokenizer, vocab2, lower = False)
    tok4 = nlp.data.BERTSPTokenizer(tokenizer, vocab4, lower = False)
    transform2 = nlp.data.BERTSentenceTransform(tok2, max_seq_length = 64, pad = True, pair= False)
    transform4 = nlp.data.BERTSentenceTransform(tok4, max_seq_length = 64, pad = True, pair= False)
    
    if not ADAPTER:
        4_model_url = "./ckpt/4way/200_ckpt.pth"
        4_onnx_name = "./4-way.onnx"
		  4_model = BertClassifier.BERTClassifier4way(bertmodel4, dr_rate=0.5)
        2_model_url = "./ckpt/nsmcAdapter128/150_ckpt.pth" 
        2_onnx_name = "./nsmc.onnx"
		  2_model = BertClassifier.BERTClassifier(bertmodel4, dr_rate=0.5) 
    if ADAPTER:
        4_model_url = "./ckpt/4wayAdapter128/150_ckpt.pth"
        4_onnx_name = "./4-way-adapter.onnx"
        4_model = BertClassifier.BERTClassifier4way(bertmodel2, dr_rate=0.5) 
        2_model_url = "./ckpt/nsmcAdapter128/150_ckpt.pth" 
        2_onnx_name = "./nsmc-adapter.onnx"
        2_model = BertClassifier.BERTClassifier(bertmodel2, dr_rate=0.5)
    
    4_model.load_state_dict(torch.load(4_model_url))
    4_model.eval()

    2_model.load_state_dict(torch.load(2_model_url))
    2_model.eval()

    4_session = onnxruntime.InferenceSession(4_onnx_name)  
    2_session = onnxruntime.InferenceSession(2_onnx_name)

    return transform2, transform4, 4_model, 2_model, 4_session, 2_session 

if __name__=="__main__":
    transform2, transform4, 4_model, 2_model, 4_session, 2_session = initsetting()
    app.run(host="127.0.0.1", port=5000, debug=True)
