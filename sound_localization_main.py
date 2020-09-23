import os,glob,sys
import time
from options.train_options import TrainOptions
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
#from localization_losses import *
from utils import *
import pdb
from Sound_Localization_Dataset import *
from network import *
from torch.utils.data import Dataset, DataLoader, RandomSampler
from losses import *
from PIL import Image
import matplotlib.pyplot as plt



epoch_logger = Logger('sound_localization_train.log',['epoch', 'loss'])

opt = TrainOptions().parse()

def overlay(img, heatmap, cmap = 'jet', alpha=0.5):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if isinstance(heatmap, np.ndarray):
        colorize = plt.get_cmap(cmap)
        #Normalize
        heatmap = heatmap - np.min(heatmap)
        heatmap = heatmap / np.max(heatmap)
        heatmap = colorize(heatmap, bytes = True)
        heatmap = Image.fromarray(heatmap[:,:,:3],mode='RGB')
    # Resize the heatmap to cover whole img
    heatmap = heatmap.resize((img.size[0], img.size[1]), resample = Image.BILINEAR)
    # Display final overlayed output
    result = Image.blend(img, heatmap, alpha)
    return result

def attention_visualization(datum_val,att_map_val,vis_folder,raw_folder):
    for i in range(len(att_map_val)):
        #Get Video Name
        sample = datum_val[i]
        video_path = sample.replace('\n','')
        words = [word.replace('\n','') for word in video_path.split('/')]
        video_name = words[-1][:-4]
        # Read the image frame of the video
        all_frames = glob.glob(video_path+'/*.jpg')
        all_frames = sorted(all_frames)
        image_path = str(all_frames[0])
        # Resize it to the network input size
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((320, 320))
        # Get the predicted attention map and reshape
        att_map_t = att_map_val[i]
        att_map = att_map_t.squeeze().detach().cpu().numpy()
        att_map = np.reshape(att_map,(20,20))
        # Overlay it onto frame
        result = overlay(image_resized, att_map)
        vis_name = video_name + '.png'
        result.save(vis_folder+'/'+vis_name)
        # Save the attention map as .npy file for accuracy calculation
        raw_val_name = video_name + '.npy'
        np.save(raw_folder+'/'+raw_val_name, att_map)


def create_optimizer(net, opt):
        if opt.optimizer == 'sgd':
                return torch.optim.SGD(net.parameters(), lr = opt.lr_rate, momentum=opt.beta1, weight_decay=opt.weight_decay)
        elif opt.optimizer == 'adam':
                return torch.optim.Adam(net.parameters(), lr = opt.lr_rate, betas=(opt.beta1,0.999), weight_decay=opt.weight_decay)


def decrease_learning_rate(optimizer, decay_factor=0.1):
        for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_factor


def evaluate(model, writer, index, dataset_val, opt):
        # CREATE FOLDER FOR EACH EPOCH
        os.makedirs(os.path.join('.', 'vis_folder', 'epoch_'+str(index)))
        os.makedirs(os.path.join('.', 'raw_folder', 'epoch_'+str(index)))
        raw_folder = os.path.join('.', 'raw_folder', 'epoch_'+str(index))
        vis_folder = os.path.join('.', 'vis_folder', 'epoch_'+str(index))
        val_losses = []
        val_unsup_losses = []
        val_sup_losses = []
        with torch.no_grad():
                for i, (frame_t_val, pos_audio_val, neg_audio_val, worker_gt_val, weights_t_val, datum_val) in enumerate(dataset_val):
                    print('Eval step:',i)
                    frame_t_val = frame_t_val.to(opt.device)
                    pos_audio_val = pos_audio_val.to(opt.device)
                    neg_audio_val = neg_audio_val.to(opt.device)
                    worker_gt_val = worker_gt_val.to(opt.device)
                    weights_t_val = weights_t_val.to(opt.device)
                    # Feed inputs into the model
                    z_val, pos_audio_embedding_val, neg_audio_embedding_val, att_map_val =  model.forward(frame_t_val, pos_audio_val, neg_audio_val)
                    # Calculate the loss
                    val_unsup_loss = unsupervised_loss_criteria(z_val, pos_audio_val, neg_audio_val, weights_t_val, opt)
                    worker_gt_val = torch.squeeze(worker_gt_val)
                    att_map_val = torch.squeeze(att_map_val)
                    val_sup_loss = supervised_loss_criteria(att_map_val, worker_gt_val, weights_t_val)
                    val_total_loss = val_unsup_loss + val_sup_loss

                    val_losses.append(val_total_loss.item())
                    val_unsup_losses.append(val_unsup_loss.item())
                    val_sup_losses.append(val_sup_loss.item())
                    # Save Attentions and overlay them on the frames
                    attention_visualization(datum_val,att_map_val,vis_folder,raw_folder)

        avg_val_loss = sum(val_losses)/len(val_losses)
        writer.add_scalar('data/val_loss', avg_val_loss, index)
        print('val loss: %.7f' % avg_val_loss)
        avg_unsup_loss = sum(val_unsup_losses)/len(val_unsup_losses)
        writer.add_scalar('data/val_unsup_loss', avg_unsup_loss, index)
        print('val unsup loss: %.7f' % avg_unsup_loss)
        avg_sup_loss = sum(val_sup_losses)/len(val_sup_losses)
        writer.add_scalar('data/val_sup_loss', avg_sup_loss, index)
        print('val sup loss: %.7f' % avg_sup_loss)
        return avg_val_loss


if opt.tensorboard=='tensorboardX':
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(comment=opt.name)
else:
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(comment=opt.name)

opt.device = torch.device("cuda") 

#Construct dataloader for train
dataset_train = Sound_Localization_Dataset(opt.dataset_file, 'train', opt.annotation_path)
dataloader_train = DataLoader(dataset_train, batch_size = opt.batchSize, num_workers= opt.nThreads, shuffle = True)

#create validation set dataloader
if opt.validation_on:
        opt.mode = 'val'
        dataset_test = Sound_Localization_Dataset(opt.val_dataset_file, 'test', opt.annotation_path)
        dataloader_test = DataLoader(dataset_test, batch_size = 32, shuffle = False, num_workers= opt.nThreads)#, sampler = RandomSampler(dataset_test,replacement=True,num_samples=100))
        dataset_size_val = len(dataloader_test)
        print('#validation audios = %d' % dataset_size_val)
        opt.mode = 'train' #set it back


model = AVModel()
#model = torch.nn.DataParallel(model, device_ids=[4,5,6,7]) ###This line was commented!
model.to(opt.device)
model.train()


# Set up optimizer
optimizer = create_optimizer(model, opt)

unsupervised_loss_criteria = UnsupervisedLoss() # margin = 1.0
supervised_loss_criteria = SupervisedLoss()

best_err = float("inf")

#pdb.set_trace()
for epoch in range(1 + opt.epoch_count, opt.niter+1):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        supervised_loss = AverageMeter()
        unsupervised_loss = AverageMeter()
        batch_loss = AverageMeter()
        end_time = time.time()

        for i, (frame_t, pos_audio, neg_audio, worker_gt, weights_t, datum) in enumerate(dataloader_train):
                data_time.update(time.time() - end_time)

                frame_t = frame_t.to(opt.device)
                pos_audio = pos_audio.to(opt.device)
                neg_audio = neg_audio.to(opt.device)
                worker_gt = worker_gt.to(opt.device)
                weights_t = weights_t.to(opt.device)
                # GET PREDICTIONS FROM THE MODEL
                z, pos_audio_embedding, neg_audio_embedding, att_map =  model.forward(frame_t, pos_audio, neg_audio)
                # Calculate the loss
                unsup_loss = unsupervised_loss_criteria(z, pos_audio, neg_audio, weights_t,opt)
                worker_gt = torch.squeeze(worker_gt)
                att_map = torch.squeeze(att_map)
                sup_loss = supervised_loss_criteria(att_map, worker_gt, weights_t)
                total_loss = unsup_loss + sup_loss

                batch_loss.update(total_loss, frame_t.size(0))
                supervised_loss.update(sup_loss, frame_t.size(0))
                unsupervised_loss.update(unsup_loss, frame_t.size(0))

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end_time)
                end_time = time.time()


                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Unsup. Loss {unsup_err.val:.4f} ({unsup_err.avg:.4f})\t'
                      'Sup. Loss {sup_err.val:.4f} ({sup_err.avg:.4f})\t'
                      'Loss {total_err.val:.7f} ({total_err.avg:.7f})\t'.format(epoch, i + 1, len(dataloader_train), batch_time=batch_time, data_time=data_time, unsup_err=unsupervised_loss, sup_err=supervised_loss ,total_err=batch_loss))


        if(epoch % opt.display_freq == 0):
                print('Display training progress at (epoch %d, total_epoch %d)' % (epoch, opt.niter))
                avg_total_loss = batch_loss.avg
                print('loss: %.7f' % avg_total_loss)
                writer.add_scalar('data/total_loss', avg_total_loss, epoch)
                avg_unsupervised_loss = unsupervised_loss.avg
                print('unsupervised loss: %.7f' % avg_unsupervised_loss)
                writer.add_scalar('data/unsupervised_loss', avg_unsupervised_loss, epoch)
                avg_supervised_loss = supervised_loss.avg
                print('supervised loss: %.7f' % avg_supervised_loss)
                writer.add_scalar('data/supervised_loss', avg_supervised_loss, epoch)
                print('end of display \n')

        if(epoch % opt.save_latest_freq == 0):
                print('saving the latest model (epoch %d, total_epoch %d)' % (epoch, opt.niter))
                torch.save(model.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'sound_localization_latest'+str(epoch)+'.pth'))

        if(epoch % opt.validation_freq == 0 and opt.validation_on):
                model.eval()
                opt.mode = 'val'
                print('Display validation results at (epoch %d, total_epoch %d)' % (epoch, opt.niter))
                val_err = evaluate(model, writer, epoch, dataloader_test, opt)
                print('end of display \n')
                model.train()
                opt.mode = 'train'
                #save the model that achieves the smallest validation error
                if val_err < best_err:
                        best_err = val_err
                        print('saving the best model (epoch %d, total_epoch %d) with validation error %.7f\n' % (epoch, opt.niter, val_err))
                        torch.save(model.state_dict(), os.path.join('.', opt.checkpoints_dir, opt.name, 'sound_localization_best.pth'))

        #epoch_logger.log({'epoch': epoch,'l1 loss': batch_reconstruction_loss.avg,'perceptual loss': 0, 'discriminative loss': 0 })

        #decrease learning rate
        # if(epoch in opt.lr_steps):
        #         decrease_learning_rate(optimizer, opt.decay_factor)
        #         print('decreased learning rate by ', opt.decay_factor)