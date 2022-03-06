from tqdm import tqdm

from utils.config import *
from models.LTHR import *

'''
Command:

python train.py -ds=reasoning -bsz=8 -hdd=128 -lr=0.001 -dr=0.2 -evalp=10 -max_neg_cnt=5 -max_depth=3

'''

early_stop = args['earlyStop']
if args['dataset'] == 'reasoning':
    from utils.utils_Ent_reasoning_for_synthetic_dataset import *
    early_stop = 'BLEU'
    # early_stop = 'MRR'
elif args['dataset'] == 'multiwoz':
    from utils.utils_Ent_reasoning_for_multiwoz import *
    early_stop = 'BLEU'
elif args['dataset'] == 'kvr':
    from utils.utils_Ent_reasoning_for_smd import *
    early_stop = 'BLEU'
else:
    print("[ERROR] You need to provide the --dataset information")

# Configure models and load data
avg_best, cnt, acc = 0.0, 0, 0.0
train, dev, test, lang, max_resp_len = prepare_data_seq(batch_size=int(args['batch']))

model = LTHR(
    lang,
    int(args['hidden']),
    int(args['hidden']),
    int(args['max_depth']),
    float(args['learn']),
    args['path'],
    float(args['drop']),
    relations_cnt=lang.n_relations,
    entities_cnt=lang.n_entities)

for epoch in range(200):
    print("Epoch:{}".format(epoch))
    # Run the train function
    pbar = tqdm(enumerate(train),total=len(train))
    for i, data in pbar:
        model.train(data, int(args['clip']), reset=(i==0))
        pbar.set_description(model.print_loss())
        # break
    if((epoch+1) % int(args['evalp']) == 0):
        acc = model.evaluate(dev, avg_best, early_stop)
        model.scheduler.step(acc)

        if(acc >= avg_best):
            avg_best = acc
            cnt = 0
        else:
            cnt += 1

        if(cnt == 80 or (acc==1.0 and early_stop==None)):
            print("Ran out of patient, early stop...")
            break


