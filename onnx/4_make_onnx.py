"""
pytorch --> ONNX
code for 4-way SA dataset
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
from tqdm.notebook import tqdm

OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

"""
parameters to decide
1. model, 2. batch_size, 3. device, 4. adapter
"""
# path of the model checkpoint
model = "./ckpt/4wayAdapter128//150_ckpt.pth"

# batch size
batch_size = 1

# choose between cpu or gpu
device = torch.device("cuda:1")
# device = torch.device('cpu')

# True when using adapter False if not
adapter = True 

# input to the model
if adapter:
    bertmodel, vocab = get_pytorch_kobert_model_adapter()
else:
    bertmodel, vocab = get_pytorch_kobert_model()
train_d = load_4way_train(vocab)
test_d = load_4way_test(vocab)

# train data
token_ids = []; valid_length = []; segment_ids = []; labels = []

for i in range(batch_size):
    token_id, val_len, segment_id = train_d[i][0]
    token_ids.append(token_id)
    valid_length.append(int(val_len))
    segment_ids.append(segment_id)

    label = train_d[i][1]
    labels.append(label)

token_ids = torch.LongTensor(token_ids).long().to(device)
valid_length = torch.LongTensor(valid_length).long().to(device)
segment_ids = torch.LongTensor(segment_ids).long().to(device)
labels = torch.LongTensor(labels).to(device)

# test data
t_token_ids = []; t_valid_length = []; t_segment_ids = []
for i in range(batch_size):
    t_token_id, t_val_len, t_segment_id = test_d[i][0]
    t_token_ids.append(t_token_id)
    t_valid_length.append(int(t_val_len))
    t_segment_ids.append(t_segment_id)

t_token_ids = torch.LongTensor(t_token_ids).long().to(device)
t_valid_length = torch.LongTensor(t_valid_length).long().to(device)
t_segment_ids = torch.LongTensor(t_segment_ids).long().to(device)


# load model
torch_model = BertClassifier.BERTClassifier4way(bertmodel, dr_rate=0.5).to(device)

# initialize the model with pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(torch.load(model, map_location = map_location))

# model to evaluation mode
torch_model.eval()
s_time = time.time()
torch_out = torch_model(t_token_ids, t_valid_length, t_segment_ids)
pytorch_time = time.time()-s_time

# export the model to ONNX
onnx_name = "./4-way-adapter.onnx" if adapter else "./4-way.onnx"

torch.onnx.export(torch_model, (t_token_ids, t_valid_length, t_segment_ids), 
                onnx_name,
                export_params = True,
                verbose = True,
                operator_export_type = OPERATOR_EXPORT_TYPE,
                input_names = ['token_ids', 'valid_length', 'segment_ids'],
                output_names = ['result'])
print("Export of torch_model.onnx complete!")

# check onnx model before obtaining model results
# onnx_model = onnx.load(onnx_name)
# onnx.checker.check_model(onnx_model)


# check the result by onnx runtime python API
ort_session = onnxruntime.InferenceSession(onnx_name)

if (device == 'cpu'):
    ort_session.set_providers(['CPUExecutionProvider'])
else:
    ort_session.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])  

inputs = {
        'token_ids': to_numpy(t_token_ids),
        'valid_length': to_numpy(t_valid_length),
        'segment_ids': to_numpy(t_segment_ids),
        }

b_time = time.time()
pred_onnx = ort_session.run(None, inputs)
onnx_time = time.time() - b_time

print("Time by onnx: {}".format(onnx_time))
print("Time by pytorch: {}".format(pytorch_time))


