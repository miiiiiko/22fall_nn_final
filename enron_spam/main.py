import os, sys, time, json
# print(os.getcwd())
# sys.path.append('./')
import ljqpy, random,pt_utils
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
# userdir = '/data1/ljq'
# import pt_utils
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
seed_everything(88)

config = {'xkey':'text', 'ykey':'label'}



class ClassifyDataset(torch.utils.data.Dataset):
    def __init__(self, data, config, label2id) -> None:
        self.data = data
        xkey, ykey = config['xkey'], config['ykey']
        self.to_tuple = lambda z:(z[xkey], label2id[z[ykey]])

    def __len__(self): return len(self.data)
    def __getitem__(self, k):
        item = self.data[k]
        return self.to_tuple(item)

def load_dataset(dr):
    trains = ljqpy.LoadJsons(os.path.join(dr, 'train.json'))
    valids = ljqpy.LoadJsons(os.path.join(dr, 'val.json'))
    id2label = ['0','1']
    label2id = {'0':0,'1':1}
    return ClassifyDataset(trains, config, label2id), \
            ClassifyDataset(valids, config, label2id), id2label

def collate_fn(xs):
    xx = tokenizer([x[0] for x in xs], return_tensors='pt', truncation=True, padding=True, max_length=maxlen)['input_ids']
    yy = torch.LongTensor([x[1] for x in xs])
    return xx, yy

# def collate_fn_pair(xs):
#     xx = tokenizer([x[0][0] for x in xs], [x[0][1] for x in xs], return_tensors='pt', truncation='only_second', padding=True, max_length=maxlen)
#     yy = torch.LongTensor([x[1] for x in xs])
#     return xx['input_ids'], xx['token_type_ids'], yy

class Classifier(nn.Module):
	def __init__(self, encoder, n_tags, cls_only=True) -> None:
		super().__init__()
		self.n_tags = n_tags
		self.encoder = encoder
		self.fc = nn.Linear(768, n_tags)
		self.cls_only = cls_only    
	def forward(self, x, seg=None):
		if seg is None: seg = torch.zeros_like(x)
		z = self.encoder(x, token_type_ids=seg).last_hidden_state
		if self.cls_only: z = z[:,0]
		out = self.fc(z)
		return out

def plot_learning_curve(record,pic_n):
    '''
    训练作图所用函数
    '''
    y1 = record[0]
    y2 = record[1]
    x1 = np.arange(1,len(y1)+1)
    # x2 = x1[::int(len(y1)/len(y2))][0:]
    x2 = [i*int(len(y1)/len(y2)) for i in range(1,len(y2)+1)]
    fig = figure(figsize = (6,4))
    ax1 = fig.add_subplot(111)
    ax1.plot(x1,y1, c = 'tab:red', label = 'train_loss')
    ax2 = ax1.twinx()
    ax2.plot(x2,y2, c='tab:cyan', label='f1')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('f1')
    plt.title('Learning curve')
    ax1.legend(loc=1)
    ax2.legend(loc=2)
    # plt.show()
    plt.savefig(pic_n)
    return

loss_func = nn.CrossEntropyLoss()


def train_func(model, ditem):
    ditem = [x.cuda() for x in ditem]
    yy = ditem[-1]
    seg = ditem[1] if len(ditem) == 3 else None
    zz = model(ditem[0], seg=seg)
    loss = loss_func(zz, yy)
    record[0].append(loss.item())
    return {'loss': loss}

def test_func(): 
    global accu
    yt, yp = [], []
    model.eval()
    with torch.no_grad():
        for ditem in dl_valid:
            seg = ditem[1].cuda() if len(ditem) == 3 else None
            xx, yy = ditem[0].cuda(), ditem[-1]
            zz = model(xx, seg=seg)
            zz = zz.detach().cpu().argmax(-1)
            for y in yy: yt.append(y.item())
            for z in zz: yp.append(z.item())
    acc,prec,rec,f1 = accuracy_score(yt,yp),precision_score(yt,yp),recall_score(yt,yp),f1_score(yt,yp)
    record[1].append(f1)
    print(f'acc: {acc:.4f}, precision: {prec:.4f}, recall: {rec: .4f}, f1: {f1: .4f}')
    model.train()
    return f1

# name = sys.argv[1]
# name = 'SST-2'

# if type(configs[name]['xkey']) is tuple: collate_fn = collate_fn_pair
# maxlen = configs[name].get('maxlen', 128)
maxlen = 256
datadir = 'dataset'

ds_train, ds_valid, id2label = load_dataset(datadir)
dl_train = torch.utils.data.DataLoader(ds_train, batch_size=32, collate_fn=collate_fn, shuffle=True)
dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size=32, collate_fn=collate_fn)

# sys.path.append('../plm_trainer/')

#################################################
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
BertModel = RobertaModel.from_pretrained('roberta-base')

model = Classifier(BertModel, len(id2label))
# model.encoder.load_state_dict(torch.load('./myroberta_v4.pt'), strict=False)
#################################################

#tokenizer = BertTokenizer.from_pretrained(plm)
#model = Classifier(BertModel.from_pretrained(plm), len(id2label))




def train_model(model, optimizer, train_dl=dl_train, epochs=3, train_func=None, test_func=None, 
                scheduler=None, save_file=None, accelerator=None, epoch_len=None):  
    '''
    模型的主训练函数
    '''
    best_f1 = -1
    for epoch in range(epochs):
        model.train()
        print(f'\nEpoch {epoch+1} / {epochs}:')
        if accelerator:
            pbar = tqdm(train_dl, total=epoch_len, disable=not accelerator.is_local_main_process)
        else: 
            pbar = tqdm(train_dl, total=epoch_len)
        metricsums = {}
        iters, accloss = 0, 0
        for ditem in pbar:
            metrics = {}
            loss = train_func(model, ditem)
            if type(loss) is type({}):
                metrics = {k:v.detach().mean().item() for k,v in loss.items() if k != 'loss'}
                loss = loss['loss']
            iters += 1; accloss += loss
            optimizer.zero_grad()
            if accelerator: 
                accelerator.backward(loss)
            else: 
                loss.backward()
            optimizer.step()
            if scheduler:
                if accelerator is None or not accelerator.optimizer_step_was_skipped:
                    scheduler.step()
            for k, v in metrics.items(): metricsums[k] = metricsums.get(k,0) + v
            infos = {'loss': f'{accloss/iters:.4f}'}
            for k, v in metricsums.items(): infos[k] = f'{v/iters:.4f}' 
            pbar.set_postfix(infos)
            if epoch_len and iters > epoch_len: break
        pbar.close()
        if test_func:
            if accelerator is None or accelerator.is_local_main_process: 
                model.eval()
                f1 = test_func()
                if f1 >=best_f1 and save_file:
                    if accelerator:
                        accelerator.wait_for_everyone()
                        unwrapped_model = accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), save_file)
                    else:
                        torch.save(model.state_dict(), save_file)
                    print(f"Epoch {epoch + 1}, best model saved. (f1: {f1:.4f})")
                    best_f1 = f1

if __name__ == '__main__':
    record = [[],[]]
    epochs = 10
    mfile = './model_states/roberta_train10.pt'
    total_steps = len(dl_train) * epochs
    optimizer, scheduler = pt_utils.get_bert_optim_and_sche(model, 1e-5, total_steps)
    model.cuda()
    train_model(model, optimizer, dl_train, epochs, train_func, test_func, scheduler=scheduler, save_file=mfile)
    plot_learning_curve(record,'./pic/roberta_train10')
