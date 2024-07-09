from model.msat_single import msat_single
#from model.dlow import DLow
import torch
from utils.config import Config

model_name = 'msat_single'
pre_path = '/home/mide/msat/results/MSAT_RESULTS/msat-nosensor-exp5/msat-nosensor-base/models/model_0030.p'
cfg = Config('msat-nosensor-base', False, create_dirs=True)
if model_name =='msat_single':
    base_model = msat_single(cfg=cfg)

else: 
    base_model = DLow(cfg=cfg)
    print('loading Dlow base model')
model_cp = torch.load(pre_path, map_location='cuda:0')
base_model.load_state_dict(model_cp['model_dict'])

cp_path = '/home/mide/msat/results/MSAT_RESULTS/msat-nosensor-exp5/msat-nosensor-base/encoders-decoders/model.p'
model_cp = {'model_dict': base_model.state_dict(), 
'context_encoder_dict':base_model.context_encoder.state_dict(),
'future_encoder_dict':base_model.future_encoder.state_dict(),
'future_decoder_dict':base_model.future_decoder.state_dict()}
torch.save(model_cp,cp_path)