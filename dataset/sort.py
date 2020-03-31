import os
import random
import pandas as pd

data_path = "./final_remove_dup_test.csv" 
csv_path = "./final_remove_dup_sort_test.csv"

train_sentence = []
train_label = []

happy = []
sad = []
anger = []
surprised = []
neutral = []

df = pd.read_csv(data_path)
label = df['label']
sentence = df['sentence']
total_num = len(sentence)

for i in range(total_num):
	if (label[i] == 0): neutral.append(sentence[i])
	elif (label[i] == 1): happy.append(sentence[i])
	elif (label[i] == 2): sad.append(sentence[i])
	elif (label[i] == 3): anger.append(sentence[i])
	elif (label[i] == 4): surprised.append(sentence[i])
	else: print("wrong emotion!!!: {}".format(label[i]))

emotion = [neutral, happy, sad, anger, surprised]
for k in range(5):
	_emotion = emotion[k]
	total_len = len(_emotion)
	print("length of emotion {} is {}".format(k, total_len))
	for m in range(total_len):
		train_sentence.append([_emotion[m], k])

_sentence = []
_label = []
random.shuffle(train_sentence)
for pair in train_sentence:
	_sentence.append(pair[0])
	_label.append(pair[1])

_df = pd.DataFrame({"sentence": _sentence, "label": _label})
_df.to_csv(csv_path)
