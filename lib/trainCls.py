import torch
import numpy as np
from utils.utils import AverageMeter, Flip
from utils.eval import AccViewCls
from utils.hmParser import parseHeatmap
import cv2
import ref
from progress.bar import Bar
from utils.debugger import Debugger

def step(split, epoch, opt, dataLoader, model, criterion, optimizer = None):
  if split == 'train':
    model.train()
  else:
    model.eval()
  preds = []
  Loss, Acc = AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('{}'.format(opt.expID), max=nIters)
  
  for i, (input, view) in enumerate(dataLoader):
    input_var = torch.autograd.Variable(input.cuda(opt.GPU, async = True)).float().cuda(opt.GPU)
    target_var = torch.autograd.Variable(view.view(-1)).long().cuda(opt.GPU)
    output = model(input_var)

    numBins = opt.numBins
    loss =  torch.nn.CrossEntropyLoss(ignore_index = numBins).cuda(opt.GPU)(output.view(-1, numBins), target_var)

    Acc.update(AccViewCls(output.data, view, numBins, opt.specificView))
    Loss.update(loss.data[0], input.size(0))

    if split == 'train':
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    else:
      if opt.test:
        out = {}
        input_ = input.cpu().numpy()
        input_[0] = Flip(input_[0]).copy()
        inputFlip_var = torch.autograd.Variable(torch.from_numpy(input_).view(1, input_.shape[1], ref.inputRes, ref.inputRes)).float().cuda(opt.GPU)
        outputFlip = model(inputFlip_var)
        pred = outputFlip.data.cpu().numpy()
        numBins = opt.numBins
        
        if opt.specificView:
          nCat = len(ref.pascalClassId)
          pred = pred.reshape(1, nCat, 3 * numBins)
          azimuth = pred[0, :, :numBins]
          elevation = pred[0, :, numBins: numBins * 2]
          rotate = pred[0, :, numBins * 2: numBins * 3]
          azimuth = azimuth[:, ::-1]
          rotate = rotate[:, ::-1]
          output_flip = []
          for c in range(nCat):
            output_flip.append(np.array([azimuth[c], elevation[c], rotate[c]]).reshape(1, numBins * 3))
          output_flip = np.array(output_flip).reshape(1, nCat * 3 * numBins)
        else:
          azimuth = pred[0][:numBins]
          elevation = pred[0][numBins: numBins * 2]
          rotate = pred[0][numBins * 2: numBins * 3]
          azimuth = azimuth[::-1]
          rotate = rotate[::-1]
          output_flip = np.array([azimuth, elevation, rotate]).reshape(1, numBins * 3)
        out['reg'] = (output.data.cpu().numpy() + output_flip) / 2.
        preds.append(out)
 
    Bar.suffix = '{split:5} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | Acc {Acc.avg:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, Acc=Acc, split = split)
    bar.next()
  bar.finish()
  return {'Loss': Loss.avg, 'Acc': Acc.avg}, preds

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)

