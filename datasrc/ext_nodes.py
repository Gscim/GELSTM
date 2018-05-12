import json
import sys
import torch
import torch.nn as nn

em_in_path = "./torch_nn64.embedding"
ent_in_path = "./train.entlist"
w2id_in_path = "./word2id.json"
out_path = "./torchnn64_h.embedding"

nnemb = torch.load(em_in_path).cpu()
with open(ent_in_path, 'r', encoding='utf-8') as entin, open(out_path, 'w', encoding='utf-8') as fbout, open(w2id_in_path, 'r') as word_2_id:
	word2id = json.load(word_2_id)
	wholeemb = {}
	entemb = []
	cnt = 0
	i = 1
	'''
	line = embin.readline()
	for line in embin:
		id_emb_ = line.strip().split()[:65]
		id_ = id_emb_[0]
		emb_ = id_emb_[1:65]
		wholeemb[id_] = emb_;

	for ent in entin:
		ent = ent.strip()
		if ent not in word2id:
			print("you sb got the code wrong!")
			sys.exit()
		if ent in word2id:
			tmid = str(word2id[ent])
			if tmid not in wholeemb:
				wholeemb[tmid] = [0.0 for n_ in range(64)]
			entemb.append(wholeemb[str(word2id[ent])])
	'''
	for ent in entin:
		ent = ent.strip()
		if ent not in word2id:
			print("error shit!!")
			sys.exit()
		if ent in word2id:
			tmid = word2id[ent]
			if tmid > 16296:
				print("somethong wrong")
			ttm = torch.LongTensor([tmid])
			entemb.append(nnemb(ttm).detach().numpy()[0])

	embn = len(entemb)
	fbout.write(str(embn) + " 64\n")
	for i in range(embn):
		fbout.write(str(i))
		for j in range(64):
			fbout.write(" " + str(entemb[i][j]))
		fbout.write("\n")



		


	

