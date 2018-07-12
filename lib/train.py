import torch
import numpy as np
from utils.utils import AverageMeter, Flip
from utils.eval import getPreds
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
  Loss, LossStar = AverageMeter(), AverageMeter()
  
  nIters = len(dataLoader)
  bar = Bar('{}'.format(opt.expID), max=nIters)
  
  for i, (input, target, mask) in enumerate(dataLoader):
    if mask.size(1) > 1:
      mask[:, 1:, :, :] *= ref.outputRes * (opt.regWeight ** 0.5)  
    if opt.GPU > -1:
      input_var = torch.autograd.Variable(input.cuda(opt.GPU, async = True)).float().cuda(opt.GPU)
      target_var = torch.autograd.Variable(target.cuda(opt.GPU, async = True)).float().cuda(opt.GPU)
      mask_var = torch.autograd.Variable(mask.cuda(opt.GPU, async = True)).float().cuda(opt.GPU)
    else:
      input_var = torch.autograd.Variable(input).float()
      target_var = torch.autograd.Variable(target).float()
      mask_var = torch.autograd.Variable(mask).float()
    output = model(input_var)
    
    output_pred = output[opt.nStack - 1].data.cpu().numpy().copy()
    for k in range(opt.nStack):
      output[k] = mask_var * output[k]
    target_var = mask_var * target_var
    
    loss = 0
    for k in range(opt.nStack):
      loss += criterion(output[k], target_var)

    LossStar.update(((target.float()[:, 0, :, :] - output[opt.nStack - 1].cpu().data.float()[:, 0, :, :]) ** 2).mean())
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
        output_flip = outputFlip[opt.nStack - 1].data.cpu().numpy()
        output_flip[0] = Flip(output_flip[0])
        if not (opt.task == 'star'):
          output_flip[0, 1, :, :] = - output_flip[0, 1, :, :]
        output_pred = (output_pred + output_flip) / 2.0
        out['map'] = output_pred
        preds.append(out)
 
    Bar.suffix = '{split:5} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | Loss {loss.avg:.6f} | LossStar {lossStar.avg:.6f}'.format(epoch, i, nIters, total=bar.elapsed_td, eta=bar.eta_td, loss=Loss, lossStar = LossStar, split = split)
    bar.next()
 
    if opt.DEBUG > 1 or (opt.DEBUG == 1 and i % (nIters / 200) == 0):
      for j in range(input.size(0)):
        debugger = Debugger()
        img = (input[j].numpy()[:3].transpose(1, 2, 0)*256).astype(np.uint8).copy()
        img2 = img.copy().astype(np.float32)
        img3 = img.copy().astype(np.float32)
        imgMNS = img.copy()
        out = (cv2.resize(((output[opt.nStack - 1][j, 0].data).cpu().numpy()).copy(), (ref.inputRes, ref.inputRes)) * 256)
        gtmap = (cv2.resize((target[j, 0].cpu().numpy()).copy(), (ref.inputRes, ref.inputRes)) * 256)
        out[out < 0] = 0
        out[out > 255] = 255
        img2[:,:,0] = (img2[:,:,0] + out)
        img2[img2 > 255] = 255
        img3[:,:,2] = (img3[:,:,2] + gtmap)
        img3[img3 > 255] = 255
        gtmap[gtmap > 255] = 255
        idx = i * input.size(0) + j if opt.DEBUG == 1 else 0
        img2, out, gtmap, img3 = img2.astype(np.uint8), out.astype(np.uint8), gtmap.astype(np.uint8), img3.astype(np.uint8)
          
        if 'emb' in opt.task:
          gt, pred = [], []
          ps = parseHeatmap(target[j].numpy())
          print('ps', ps)
          for k in range(len(ps[0])):
            print('target', k, target[j, 1:4, ps[0][k], ps[1][k]].numpy())
            x, y, z = ((target[j, 1:4, ps[0][k], ps[1][k]].numpy() + 0.5) * 255).astype(np.int32)
            gt.append(target[j, 1:4, ps[0][k], ps[1][k]].numpy())
            cv2.circle(imgMNS, (ps[1][k] * 4, ps[0][k] * 4), 6, (int(x), int(y), int(z)), -1)
            
          ps = parseHeatmap(output_pred[j])
          for k in range(len(ps[0])):
            print('pred', k, output_pred[j, 1:4, ps[0][k], ps[1][k]])
            x, y, z = ((output_pred[j, 1:4, ps[0][k], ps[1][k]] + 0.5) * 255).astype(np.int32)
            pred.append(output_pred[j, 1:4, ps[0][k], ps[1][k]])
            cv2.circle(imgMNS, (ps[1][k] * 4, ps[0][k] * 4), 4, (255, 255, 255), -1)
            cv2.circle(imgMNS, (ps[1][k] * 4, ps[0][k] * 4), 2, (int(x), int(y), int(z)), -1)
          debugger.addPoint3D(np.array(gt), c = 'auto', marker = 'o')
          #debugger.addPoint3D(np.array(pred), c = 'auto', marker = 'x')
        debugger.addImg(imgMNS, '{}_mns'.format(idx))
        debugger.addImg(out, '{}_out'.format(idx))
        debugger.addImg(gtmap, '{}_gt'.format(idx))
        debugger.addImg(img, '{}_img'.format(idx))
        debugger.addImg(img2, '{}_img2'.format(idx))
        debugger.addImg(img3, '{}_img3'.format(idx))
        if opt.DEBUG == 1:
          debugger.saveAllImg(path = opt.debugPath)
        else:
          debugger.showAllImg(pause = not('emb' in opt.task))
        if 'emb' in opt.task:
          debugger.show3D()
 
  bar.finish()
  return {'Loss': Loss.avg, 'LossStar': LossStar.avg}, preds
  

def train(epoch, opt, train_loader, model, criterion, optimizer):
  return step('train', epoch, opt, train_loader, model, criterion, optimizer)
  
def val(epoch, opt, val_loader, model, criterion):
  return step('val', epoch, opt, val_loader, model, criterion)

