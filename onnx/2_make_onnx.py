"""
pytorch --> ONNX
code for NSMC dataset
"""
import io
import numpy as np
import time
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import onnx
import onnxruntime
from Models import BertClassifier
from load_dataset import *
from KoBERT.kobert.pytorch_kobert_adapter import get_pytorch_kobert_model_adapter
from KoBERT.kobert.pytorch_kobert import get_pytorch_kobert_model

OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

"""
parameters to decide
1. model, 2. batch_size, 3.device, 4. adapter
"""
# path of the model checkpoint
model = "./ckpt/nsmcAdapter128/50_ckpt.pth"

# batch size
batch_size = 1

# choose between cpu or gpu
# device = torch.device("cuda:1")
device = 'cpu'

# True when using adapter False if not
adapter = True 

# input to the model 
if adapter:
    bertmodel, vocab = get_pytorch_kobert_model_adapter()
else:
    bertmodel, vocab = get_pytorch_kobert_model()
train_d = load_nsmc_train_part(vocab)
test_d = load_nsmc_test(vocab)

# train data
token_ids = []; valid_length = []; segment_ids = [];

for i in range(batch_size):
    token_id, val_len, segment_id = train_d[i][0]
    token_ids.append(token_id)
    valid_length.append(int(val_len))
    segment_ids.append(segment_id)

token_ids = torch.LongTensor(token_ids).long().to(device)
valid_length = torch.LongTensor(valid_length).long().to(device)
segment_ids = torch.LongTensor(segment_ids).long().to(device)

# test data
t_token_ids = []; t_valid_length = []; t_segment_ids = [];
for i in range(batch_size):
    t_token_id, t_val_len, t_segment_id = test_d[0][0]
    t_token_ids.append(t_token_id)
    t_valid_length.append(int(t_val_len))
    t_segment_ids.append(t_segment_id)

t_token_ids = torch.LongTensor(t_token_ids).long().to(device)
t_valid_length = torch.LongTensor(t_valid_length).long().to(device)
t_segment_ids = torch.LongTensor(t_segment_ids).long().to(device)

# load model
torch_model = BertClassifier.BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# initialize the model with pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(torch.load(model))

# model to evaluation mode
torch_model.eval()
s_time = time.time()
torch_out = torch_model(token_ids, valid_length, segment_ids)
pytorch_time = time.time()-s_time

# export the model to ONNX
onnx_name = "./nsmc.onnx" if not adapter else "./nsmc-adapter.onnx"
torch.onnx.export(torch_model, (token_ids, valid_length, segment_ids), 
                onnx_name,
                verbose = True,
                operator_export_type = OPERATOR_EXPORT_TYPE,
                input_names = ['token_ids', 'valid_length', 'segment_ids'])
print("Export of torch_model.onnx complete!")

# check onnx model before obtaining model results
# onnx_model = onnx.load(onnx_name)
# onnx.checker.check_model(onnx_model)

# check the result by onnx runtime python API
ort_session = onnxruntime.InferenceSession(onnx_name)

if (device="cpu"):
	ort_session.set_providers(['CPUExecutionProvider'])
else:
	ort_session.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])

b_time = time.time()
pred_onnx = ort_session.run(None, {'token_ids': to_numpy(t_token_ids),
                            'valid_length': to_numpy(t_valid_length),
                            'segment_ids': to_numpy(t_segment_ids)})
print("Time by onnx: {}".format(time.time()-b_time))
print("Time by pytorch: {}".format(pytorch_time))

