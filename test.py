from utils.config import *
from models.LTHR import *

'''
Command:

python test.py -ds=reasoning -bsz=8 -hdd=128 -lr=0.001 -dr=0.2 -evalp=10 -max_neg_cnt=5 -max_depth=3 -path=save/LTHR-DialogueReasoning/HDD128BSZ8lr0.001MRR-0.9722

'''

directory = args['path'].split("/")
HDD = directory[2].split('HDD')[1].split('BSZ')[0]
BSZ = int(directory[2].split('BSZ')[1].split('lr')[0])
if args['dataset'] == 'reasoning':
    DS = 'reasoning'
elif args['dataset'] == 'multiwoz':
    DS = 'multiwoz'
elif args['dataset'] == 'kvr':
    DS = 'kvr'

if DS == 'reasoning':
    from utils.utils_Ent_reasoning_for_synthetic_dataset import *
elif DS == 'multiwoz':
    from utils.utils_Ent_reasoning_for_multiwoz import *
elif DS == 'kvr':
    from utils.utils_Ent_reasoning_for_smd import *
else:
    print("You need to provide the --dataset information")

train, dev, test, lang, max_resp_len = prepare_data_seq(batch_size=BSZ)

model = LTHR(
	lang,
    int(HDD),
    int(HDD),
    int(args['max_depth']),
    float(args['learn']),
	args['path'],
    float(args['drop']),
    relations_cnt=lang.n_relations,
    entities_cnt=lang.n_entities)

acc_test = model.evaluate(test, 1e7)
