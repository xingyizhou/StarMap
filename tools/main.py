from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import time
import datetime
from utils.logger import Logger
import torch
import torch.utils.data
from opts import opts
import ref
from model import getModel, saveModel
opt = opts().parse()

if opt.task == 'cls':
  from datasets.Pascal3DCls import Pascal3D as Dataset
  from trainCls import train, val
else:
  if opt.dataset == 'Pascal3D':
    from datasets.Pascal3D import Pascal3D as Dataset
  elif opt.dataset == 'ObjectNet3D':
    from datasets.ObjectNet3D import ObjectNet3D as Dataset
  else:
    raise(Exception('Dataset Not Exists!'))
  from train import train, val

def main():
  now = datetime.datetime.now()
  logger = Logger(opt.saveDir + '/logs_{}'.format(now.isoformat()))
  model, optimizer = getModel(opt)

  criterion = torch.nn.MSELoss()
  
  if opt.GPU > -1:
    print('Using GPU', opt.GPU)
    model = model.cuda(opt.GPU)
    criterion = criterion.cuda(opt.GPU)
  
  val_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'val'), 
      batch_size = 1, 
      shuffle = True if opt.DEBUG > 1 else False,
      num_workers = 1
  )

  if opt.test:
    _, preds = val(0, opt, val_loader, model, criterion)
    torch.save({'opt': opt, 'preds': preds}, os.path.join(opt.saveDir, 'preds.pth'))
    return

  train_loader = torch.utils.data.DataLoader(
      Dataset(opt, 'train'), 
      batch_size = opt.trainBatch, 
      shuffle = True,
      num_workers = int(opt.nThreads)
  )

  for epoch in range(1, opt.nEpochs + 1):
    mark = epoch if opt.saveAllModels else 'last'
    log_dict_train, _ = train(epoch, opt, train_loader, model, criterion, optimizer)
    for k, v in log_dict_train.items():
      logger.scalar_summary('train_{}'.format(k), v, epoch)
      logger.write('{} {:8f} | '.format(k, v))
    if epoch % opt.valIntervals == 0:
      log_dict_val, preds = val(epoch, opt, val_loader, model, criterion)
      for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      saveModel(os.path.join(opt.saveDir, 'model_{}.checkpoint'.format(mark)), model) # optimizer
    logger.write('\n')
    if epoch % opt.dropLR == 0:
      lr = opt.LR * (0.1 ** (epoch // opt.dropLR))
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr
  logger.close()
  torch.save(model.cpu(), os.path.join(opt.saveDir, 'model_cpu.pth'))

if __name__ == '__main__':
  main()
