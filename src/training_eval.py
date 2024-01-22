import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics


def model_initializer(checkpoint_path, model):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint['model_state_dict'])
  return model

def embedding_phase(dataloaders, phase, model):

  print('embdedding phase')
  labels = []
  embeddings = []

  for count, inputs in enumerate(dataloaders[phase]):

    labels.append(np.squeeze(inputs['label'].cpu().detach().numpy()))
    #labels.append(target[inputs['label'].cpu().detach()])
    try:
      embeddings.append(np.squeeze(model(inputs).cpu().detach().numpy()))
      return labels, embeddings
    except:
      print('error')

# From pred before ensamble
'''def eval_model1_model2(model1, model2, dataloaders, device):
  origin = []
  pred = []
  misclassified = []
  
  model1.eval()
  for count, inputs in enumerate(dataloaders["test"]):
    labels = inputs['label'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    outputs1 = model1(inputs)
    outputs2 = model2(inputs)
    outputs = outputs1 + outputs2
    _, preds = torch.max(outputs, 1)
    pred.extend(preds.cpu().detach().numpy())

    if preds.cpu().detach().numpy() != labels.cpu().detach().numpy():
      misclassified.append([inputs['name'][0], inputs['target'][0], labels.cpu().numpy()[0], preds.cpu().detach().numpy()[0]])

  return origin, pred'''

# From bootstrap script
'''
def plot_train_test(train_list, test_list, title, label1, label2, level = None):
  epochs = [i  for i in range(args.epoch_number)]
  
  fig, ax = plt.subplots(figsize = (5, 2.7), layout = 'constrained')
  ax.plot(epochs, train_list, label = label1)
  ax.plot(epochs, test_list, label = label2)

  if level is not None:
    ax.plot(epochs, [level]*args.epoch_number, label = 'max')
    ax.plot(epochs, [level//2]*args.epoch_number, label = 'threshold')
    ax.plot(epochs, [0]*args.epoch_number, label = 'min')
    ax.set_ylabel('Classes')
  else:
    ax.set_ylabel(args.criterion)

  ax.set_xlabel('Epoch')
  #ax.set_ylabel(args.criterion)
  ax.set_title(title)
  ax.legend()

  image_save_path = os.path.join(args.checkpoint,'figures',args.save_name)

  if not os.path.exists(image_save_path):
            os.mkdir(image_save_path)

  # Save figure
  plt.savefig(image_save_path+'/'+title +'.png')

def eval_model(model, dataloaders):
  origin = []
  pred = []
  misclassified =[]
  
  model.eval()
  for count, inputs in enumerate(dataloaders["test"]):
    labels = inputs['label'].to(device)
    if args.smote:
      inputs = inputs['X'].to(device)
    origin.extend(labels.cpu().detach().numpy())
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    pred.extend(preds.cpu().detach().numpy())
    #if preds.cpu().detach().numpy() != labels.cpu().detach().numpy():
    #  misclassified.append([inputs['name'][0], inputs['target'][0], labels.cpu().numpy()[0], preds.cpu().detach().numpy()[0]])
    
  #print(misclassified)

  confusion_matrix = metrics.confusion_matrix(np.asarray(origin), np.asarray(pred))
  cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
  cm_display.plot()
  plt.show()
  #plt.matshow(confusion_matrix)
  #plt.title('Confusion Matrix')
  #plt.colorbar()
  #plt.ylabel('True Label')
  #plt.xlabel('Predicated Label')
  plt.savefig('confusion_matrix.jpg')


  return origin, pred
'''

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
        print('Here')
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
  


def eval_model1_model2_smote(model1, model2, dataloader1, dataloader2, device, misclassified_flag = False, confusion_matrix_flag = False):
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