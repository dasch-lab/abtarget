import torch
import os
import time
import copy

from torchmetrics.classification import BinaryF1Score
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score
from matplotlib import pyplot as plt
from sklearn import metrics
import numpy as np

from src.metrics import MCC



def model_initializer(checkpoint_path, model):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  return model

def embedding_phase(dataloaders, phase, model):

  print('embdedding phase')
  labels = []
  embeddings = []

  '''target =  {'peptide | peptide | peptide':0, 'peptide | protein | protein':1, 'peptide | protein':2, 'protein':3, 'protein | peptide':4, 'protein | protein | protein | protein':5, 
                  'protein | peptide | protein':6, 'protein | protein':7, 'protein | protein | protein':8, 'peptide | peptide':9, 'peptide':10, 'protein | protein | protein | peptide':11,
                  'protein | protein | protein | protein | protein':12, 'protein | protein | peptide':13,'Hapten':14, 'carbohydrate':15, 'nucleic-acid':16, 'nucleic-acid | nucleic-acid | nucleic-acid':17, 'nucleic-acid | nucleic-acid':18}'''
  

  for count, inputs in enumerate(dataloaders[phase]):

    labels.append(np.squeeze(inputs['label'].cpu().detach().numpy()))
    try:
      embeddings.append(np.squeeze(model(inputs).cpu().detach().numpy()))
      return labels, embeddings
    except:
      print('error')

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=1, save_folder=None, batch_size=8, device='cpu', save = True, smote = False, ensemble = False, model_name = 'antiberty', save_name = '', subset = 0, epoch_number = 0, path = ''):
  since = time.time()
  best_model_wts = copy.deepcopy(model.state_dict())
  best_acc = 0.0
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []

  for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("#" * 10)

    # Epoch train and validation phase
    for phase in ["train", "test"]:
      print("## " + f"{phase}".capitalize())
      if phase == "train":
        model.train()
      else:
        model.eval()

      actual = []
      pred = []

      # Iterate over the Data
      running_loss = 0.0
      running_correct = 0
      dataset_size = len(dataloaders[phase].dataset)
      size = len(dataloaders[phase])
      mcc_score = MCC()
      zeros = 0
      ones = 0

      print(len(dataloaders[phase]))
      try:
        for count, inputs in enumerate(dataloaders[phase]):

          labels = inputs['label'].to(device)
          if smote:
            inputs = inputs['X'].to(device)
          actual.extend(labels.tolist())
          optimizer.zero_grad()
          with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            pred.extend(preds.cpu().detach().numpy())
            one = torch.sum(preds).item()
            ones += one
            zeros += (batch_size - one)
            loss = criterion(outputs, labels)
            if phase == "train":
              loss.backward()
              optimizer.step()

          # Stats
          current_loss = loss.item()
          running_loss += current_loss * len(labels)
          running_correct += torch.sum(preds.view(-1) == labels.view(-1))
          '''if phase == "eval":
            scheduler.step(running_loss)'''
          
          #if phase == "train":
          #  scheduler.step()

          #weights = [zeros / dataset_size, ones / dataset_size]
          epoch_loss = running_loss / dataset_size
          #epoch_acc = running_correct.double() / dataset_size
          epoch_acc = balanced_accuracy_score(actual, pred)
          mcc = mcc_score.update(preds, labels)
          #epoch_f1 = f1_score(labels.cpu().numpy(), preds.cpu().numpy())
          metric = BinaryF1Score().to(device)
          epoch_f1 = metric(preds, labels)
          #isPrint = True if count % 10 == 0 or count == size-1 else False
          isPrint = True if count == size-1 else False
          if isPrint:
            print('{phase} {count}/{total} Loss: {loss:.4f} Running Loss: {running_loss:.4f} Acc: {acc:.4f} MCC: {mcc:.4f} F1: {f1:.4f}'.format(
              total=size,
              count=count,
              phase=phase,
              running_loss=epoch_loss,
              loss=current_loss,
              acc=epoch_acc,
              mcc=mcc,
              f1 = epoch_f1
            ))

          # Deep copy the model & save checkpoint to file
          if phase == "test": 

            name = ''
          
            if epoch_acc > best_acc:

              best_acc = epoch_acc
              best_model_wts = copy.deepcopy(model.state_dict())
              name = '_best_accuracy'

            if ensemble:
              save_path = os.path.join(save_folder, 'checkpoints', model_name, 'ensemble', str(subset))
            else:
              save_path = os.path.join(save_folder, 'checkpoints', model_name, 'single')
            
            if not os.path.exists(save_path):
              os.mkdir(save_path)
            
            checkpoint_path = os.path.join(save_path, save_name + name)
            if save:
              torch.save({
              "epoch": epoch,
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "loss": loss,
              "batch_size": batch_size,
              }, checkpoint_path)

        if phase == "train":
          train_loss.append(epoch_loss)
          train_acc.append(epoch_acc.item())
        else:
          test_loss.append(epoch_loss)
          test_acc.append(epoch_acc.item())
      except:
        print('error')

  # Store checkpoint
  
  checkpoint_path = os.path.join(save_path, 'epoch_{0}'.format(epoch+1))

  if save:
    torch.save({
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": loss,
      "batch_size": batch_size,
    }, checkpoint_path)

  time_elapsed = time.time() - since
  print('Training complete in {h}:{m}:{s}'.format(
    h=int(time_elapsed // 3600),
    m=int(time_elapsed // 60),
    s=int(time_elapsed % 60)
  ))
  print("Best test Acc: {0}".format(best_acc))

  # Load best model weights
  model.load_state_dict(best_model_wts)

  plot_train_test(train_loss, test_loss, 'Loss', 'train', 'val', epoch_number, criterion, path, 'Loss' )
  plot_train_test(train_acc, test_acc, 'Accuracy', 'train', 'val', epoch_number, criterion, path, 'Accuracy')

  return model, checkpoint_path

def plot_train_test(train_list, test_list, title, label1, label2, epoch_number, criterion, path, save_name, level = None):
  epochs = [i  for i in range(epoch_number)]
  
  fig, ax = plt.subplots(figsize = (5, 2.7), layout = 'constrained')
  ax.plot(epochs, train_list, label = label1)
  ax.plot(epochs, test_list, label = label2)

  if level is not None:
    ax.plot(epochs, [level]*epoch_number, label = 'max')
    ax.plot(epochs, [level//2]*epoch_number, label = 'threshold')
    ax.plot(epochs, [0]*epoch_number, label = 'min')
    ax.set_ylabel('Classes')
  else:
    ax.set_ylabel(criterion)

  ax.set_xlabel('Epoch')
  #ax.set_ylabel(args.criterion)
  ax.set_title(title)
  ax.legend()

  image_save_path = os.path.join(path,'figures',save_name)

  if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

  # Save figure
  plt.savefig(image_save_path+'/'+title +'.png')


def eval_model1_model2(dataloaders, device, single = True, list_model1=None, list_model2 = None):
  origin = []
  pred = []
  misclassified =[]

  for count, inputs in enumerate(dataloaders["test"]):
    labels = inputs['label'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    list_pred = []
    for i in range(len(list_model1)):
      if single:
        _, output = torch.max(list_model1[i](inputs), 1)
        list_pred.append(output.cpu().numpy()[0])
      else:
        output1 = list_model1[i](inputs)
        output2 = list_model2[i](inputs)
        output = output1 + output2
        _, output = torch.max(output, 1)
        list_pred.append(output.cpu().numpy()[0])

    max_eval = 0 if list_pred.count(0) > 9 else 1
    pred.append(max_eval)
    if max_eval != labels:
      misclassified.append([inputs['name'][0], inputs['target'][0], labels.cpu().numpy()[0], max_eval])
  
  print(misclassified)
  return origin, pred
  


def eval_model1_model2_smote(model1, model2, dataloader1, dataloader2, device):
  origin = []
  pred = []
  misclassified = []
  
  for count, (input1, input2) in enumerate(zip(dataloader1["test"], dataloader2["test"])):
    labels = input1['label'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    outputs1 = model1(input1['X'].to(device))
    outputs2 = model2(input2['X'].to(device))
    outputs = outputs1 + outputs2
    _, preds = torch.max(outputs, 1)
    pred.extend(preds.cpu().detach().numpy())
    
    if preds.cpu().detach().numpy() != labels.cpu().detach().numpy():
      misclassified.append([input1['name'][0], input1['target'][0], labels.cpu().numpy()[0], preds.cpu().detach().numpy()[0]])
  return origin, pred


def confusion_matrix(origin, pred):
  confusion_matrix = metrics.confusion_matrix(np.asarray(origin), np.asarray(pred))
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
  cm_display.plot()
  plt.show()
  plt.savefig('confusion_matrix.jpg')


def final_score_eval(org, pred):
  precision = metrics.precision_score(org, pred)
  recall = metrics.recall_score(org, pred)
  f1 = metrics.f1_score(org, pred)
  accuracy = metrics.accuracy_score(org, pred)
  mcc = metrics.matthews_corrcoef(org, pred)
  print('Precision: ', precision)
  print('Recall: ', recall)
  print('F1: ', f1)
  print('Accuracy: ', accuracy)
  print('MCC: ', mcc)