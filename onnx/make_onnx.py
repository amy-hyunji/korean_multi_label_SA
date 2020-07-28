"""
pytorch --> ONNX
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
from temp_load_dataset import *
from KoBERT.kobert.pytorch_kobert_adapter import get_pytorch_kobert_model_adapter

OPERATOR_EXPORT_TYPE = torch._C._onnx.OperatorExportTypes.ONNX

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def make_position_input(input_ids):
    seq_length = input_ids.size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
    return position_ids

def make_train_dummy_input():
    org_input_ids = torch.LongTensor([[31, 51, 98, 1]])
    org_token_type_ids = torch.LongTensor([[1, 1, 1, 1]])
    org_input_mask = torch.LongTensor([[0, 0, 1, 1]])
    org_position_ids = make_position_input(org_input_ids)
    return (org_input_ids, org_token_type_ids, org_input_mask, org_position_ids)

def make_inference_dummy_input():
    inf_input_ids = [[31, 51, 98]]
    inf_token_type_ids = [[1, 1, 1]]
    inf_input_mask = [[0, 0, 1]]
    inf_position_ids = range(0, len(inf_input_ids[0]))
    return(inf_input_ids, inf_token_type_ids, inf_input_mask, inf_position_ids)

model_url = "./ckpt/50_ckpt.pth"
batch_size = 4
device = torch.device("cuda:1")
# device = 'cpu'


# input to the model 
bertmodel, vocab = get_pytorch_kobert_model_adapter()
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
bertmodel, vocab = get_pytorch_kobert_model_adapter()
torch_model = BertClassifier.BERTClassifier(bertmodel, dr_rate=0.5).to(device)

# initialize the model with pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(torch.load(model_url))

# model to evaluation mode
torch_model.eval()
s_time = time.time()
torch_out = torch_model(token_ids, valid_length, segment_ids)
pytorch_time = time.time()-s_time

# export the model to ONNX
torch.onnx.export(torch_model, (token_ids, valid_length, segment_ids), 
                "./nsmc.onnx",
                verbose = True,
                operator_export_type = OPERATOR_EXPORT_TYPE,
                input_names = ['token_ids', 'valid_length', 'segment_ids'])
print("Export of torch_model.onnx complete!")

# check onnx model before obtaining model results
onnx_model = onnx.load('nsmc.onnx')
# onnx.checker.check_model(onnx_model)

# check the result by onnx runtime python API
ort_session = onnxruntime.InferenceSession("./nsmc.onnx")
if (device == 'cpu'):
    ort_session.set_providers(['CPUExecutionProvider'])
else:
    ort_session.set_providers(['CUDAExecutionProvider'])
b_time = time.time()
pred_onnx = ort_session.run(None, {'token_ids': to_numpy(t_token_ids),
                            'valid_length': to_numpy(t_valid_length),
                            'segment_ids': to_numpy(t_segment_ids)})
print("Time by onnx: {}".format(time.time()-b_time))
print("Time by pytorch: {}".format(pytorch_time))

# compare the results
"""
np.testing.assert_allclose(to_numpy(torch_out), pred_onnx[0][0:4], rtol=1e-03, atol=1e-05)

print("Done testing with ONNXRuntime! Seems like there's no problem :)")
"""
