import os
import random
import pandas as pd

def check_num(remove_neutral = True):
	single = "./original/korean_single_turn_utterance.xlsx"
	multi = "./original/korean_multi_turn_utterance.xlsx"
	naver = "./original/naver_movie_rating.xlsx"

	singleDF = pd.read_excel(single)
	multiDF = pd.read_excel(multi)
	naverDF = pd.read_excel(naver)

	singleSEN = singleDF['Sentence'].tolist()
	singleLAB = singleDF['Emotion'].tolist()
	singleLAB = change_label(singleLAB)
	totalPAIR = make_pair(singleSEN, singleLAB, remove_neutral) 

	multiSEN = multiDF['Sentence'].tolist()
	multiLAB = multiDF['Emotion'].tolist()
	multiLAB = change_label(multiLAB)
	totalPAIR += make_pair(multiSEN, multiLAB, remove_neutral)

	naverSEN = naverDF['original'].tolist()
	naverLAB = naverDF['label'].tolist()
	naverLAB = change_label(naverLAB)
	totalPAIR += make_pair(naverSEN, naverLAB, remove_neutral)

	random.shuffle(totalPAIR)
	sentence = []
	label = []
	for sen, lab in totalPAIR:
		sentence.append(sen)
		label.append(lab)
	
	output_df = pd.DataFrame({"sentence": sentence, "label": label})
	output_df.to_csv("./final_before_cleansing_no_neutral.csv")


def make_pair(sen, lab, remove):
	pair = []
	for _sen, _lab in zip(sen, lab):
		if (remove and _lab == 0):
			continue
		else:
			pair.append([_sen, _lab])
	return pair

def change_label(labelList):
	returnLAB = []
	for label in labelList:
		if (label == 0 or label == "N" or label == "중립"):
			returnLAB.append(0)
		elif (label == 1 or label == "H" or label == "행복"):
			returnLAB.append(1)
		elif (label == 2 or label == "S" or label == "슬픔"):
			returnLAB.append(2)
		elif (label == 3 or label == "A" or label == "분노" or label == "혐오"):
			returnLAB.append(3)
		elif (label == 4 or label == "U" or label == "놀람"):
			returnLAB.append(4)
		elif (label == 5 or label == "F" or label == "공포"):
			returnLAB.append(5)
	return returnLAB

def check_labelnum(name, labelList):
	labelDict = {"neutral": 0, "happy": 0, "sad": 0, "angry": 0, "surprised": 0, "fear": 0}
	for label in labelList:
		if (label == 0 or label == "N" or label =="중립"):
			labelDict['neutral'] += 1
		elif (label == 1 or label == "H" or label == "행복"):
			labelDict['happy'] += 1
		elif (label == 2 or label == "S" or label == "슬픔"):
			labelDict['sad'] += 1
		elif (label == 3 or label == "A" or label == "분노" or label == "혐오"):
			labelDict['angry'] += 1
		elif (label == 4 or label == "U" or label == "놀람"):
			labelDict['surprised'] += 1
		elif (label == 5 or label == "F" or label == "공포"):
			labelDict['fear'] += 1

	print("[{}]: neutral {}, happy: {}, sad: {}, angry: {}, surprised: {}, fear: {}".format(name, labelDict['neutral'], labelDict['happy'], labelDict['sad'], labelDict['angry'], labelDict['surprised'], labelDict['fear']))

def remove_dup():
	inputFile = "./final_before_cleansing_no_neutral.csv"
	inputDF = pd.read_csv(inputFile)
	
	df_sentence = inputDF['sentence'].tolist()
	df_label = inputDF['label'].tolist()
	print("INITIAL sentence: {}".format(len(df_sentence)))

	sentence_set = list(set(df_sentence))
	print("SET sentence: {}".format(len(sentence_set)))

	sentence = []; label = []
	for sen, lab in zip(df_sentence, df_label):
		if sen in sentence_set:
			sentence.append(sen)
			label.append(lab)
			sentence_set.remove(sen)
		else:
			continue

	print("final sentence: {}, label: {}".format(len(sentence), len(label)))
	output_df = pd.DataFrame({"sentence": sentence, "label": label})
	output_df.to_csv("./final_before_cleansing_no_neutral.csv")

if __name__ == "__main__":
	remove_dup()
