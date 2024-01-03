import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Importing the libraries needed
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertModel, DistilBertTokenizer


from utils import MyDataset, DistillBERTClass, calcuate_accu, get_train_test_data


from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'



train_dataset, test_dataset = get_train_test_data(data_path= '/home/patidarritesh/smartsense/Car_category_classification/data/data_sheet.csv')



# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 3
VALID_BATCH_SIZE = 2
EPOCHS = 15
LEARNING_RATE = 3e-04
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

training_set = MyDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = MyDataset(test_dataset, tokenizer, MAX_LEN)

print("training_set[0].keys()", training_set[0].keys())


train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# print the len of the training and testing loader
print("len(training_loader)", len(training_loader))
print("-"*100)

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.



model = DistillBERTClass(base_model=DistilBertModel.from_pretrained('distilbert-base-cased'))
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask)

        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accu(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)

        if _%4==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Training Loss per 4 steps: {loss_step}")
            print(f"Training Accuracy per 4 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return

def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples=0
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            # shape of output is [4], reshape to [1,4]
            if len(outputs.shape) == 1:
                outputs = outputs.reshape(1,4)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples
            print(f"Validation Loss per steps: {loss_step}")
            print(f"Validation Accuracy per steps: {accu_step}")
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_accu


for epoch in range(EPOCHS):
    train(epoch)


print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

acc = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % acc)


os.makedirs('./models', exist_ok=True)

# Saving the files for re-use

output_model_file = './models/pytorch_distilbert_news.bin'
output_vocab_file = './models/vocab_distilbert_news.bin'

model_to_save = model
torch.save(model_to_save, output_model_file)
tokenizer.save_vocabulary(output_vocab_file)

print('All files saved')


# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# # Importing the libraries needed
# import pandas as pd
# import torch
# import transformers
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertModel, DistilBertTokenizer


# from utils import MyDataset, DistillBERTClass, calcuate_accu, get_train_test_data


# from torch import cuda
# device = 'cuda' if cuda.is_available() else 'cpu'



# train_dataset, test_dataset = get_train_test_data(data_path= '/home/patidarritesh/smartsense/Car_category_classification/data/data_sheet.csv')



# # Defining some key variables that will be used later on in the training
# MAX_LEN = 512
# TRAIN_BATCH_SIZE = 4
# VALID_BATCH_SIZE = 2
# EPOCHS = 1
# LEARNING_RATE = 3e-04
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

# training_set = MyDataset(train_dataset, tokenizer, MAX_LEN)
# testing_set = MyDataset(test_dataset, tokenizer, MAX_LEN)

# print("training_set[0].keys()", training_set[0].keys())


# train_params = {'batch_size': TRAIN_BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }

# test_params = {'batch_size': VALID_BATCH_SIZE,
#                 'shuffle': True,
#                 'num_workers': 0
#                 }

# training_loader = DataLoader(training_set, **train_params)
# testing_loader = DataLoader(testing_set, **test_params)


# # Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.



# model = DistillBERTClass(base_model=DistilBertModel.from_pretrained('distilbert-base-cased'))
# model.to(device)

# # Creating the loss function and optimizer
# loss_function = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# # Defining the training function on the 80% of the dataset for tuning the distilbert model

# def train(epoch):
#     tr_loss = 0
#     n_correct = 0
#     nb_tr_steps = 0
#     nb_tr_examples = 0
#     model.train()
#     for _,data in enumerate(training_loader, 0):
#         ids = data['ids'].to(device, dtype = torch.long)
#         mask = data['mask'].to(device, dtype = torch.long)
#         targets = data['targets'].to(device, dtype = torch.long)

#         outputs = model(ids, mask)

#         loss = loss_function(outputs, targets)
#         tr_loss += loss.item()
#         big_val, big_idx = torch.max(outputs.data, dim=1)
#         n_correct += calcuate_accu(big_idx, targets)

#         nb_tr_steps += 1
#         nb_tr_examples+=targets.size(0)

#         if _%5000==0:
#             loss_step = tr_loss/nb_tr_steps
#             accu_step = (n_correct*100)/nb_tr_examples
#             print(f"Training Loss per 5000 steps: {loss_step}")
#             print(f"Training Accuracy per 5000 steps: {accu_step}")

#         optimizer.zero_grad()
#         loss.backward()
#         # # When using GPU
#         optimizer.step()

#     print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
#     epoch_loss = tr_loss/nb_tr_steps
#     epoch_accu = (n_correct*100)/nb_tr_examples
#     print(f"Training Loss Epoch: {epoch_loss}")
#     print(f"Training Accuracy Epoch: {epoch_accu}")

#     return

# def valid(model, testing_loader):
#     model.eval()
#     n_correct = 0; n_wrong = 0; total = 0
#     tr_loss = 0
#     nb_tr_steps = 0
#     nb_tr_examples=0
#     with torch.no_grad():
#         for _, data in enumerate(testing_loader, 0):
#             ids = data['ids'].to(device, dtype = torch.long)
#             mask = data['mask'].to(device, dtype = torch.long)
#             targets = data['targets'].to(device, dtype = torch.long)
#             outputs = model(ids, mask).squeeze()
#             # shape of output is [4], reshape to [1,4]
#             if len(outputs.shape) == 1:
#                 outputs = outputs.reshape(1,4)
#             loss = loss_function(outputs, targets)
#             tr_loss += loss.item()
#             big_val, big_idx = torch.max(outputs.data, dim=1)
#             n_correct += calcuate_accu(big_idx, targets)

#             nb_tr_steps += 1
#             nb_tr_examples+=targets.size(0)

#             if _%5000==0:
#                 loss_step = tr_loss/nb_tr_steps
#                 accu_step = (n_correct*100)/nb_tr_examples
#                 print(f"Validation Loss per 100 steps: {loss_step}")
#                 print(f"Validation Accuracy per 100 steps: {accu_step}")
#     epoch_loss = tr_loss/nb_tr_steps
#     epoch_accu = (n_correct*100)/nb_tr_examples
#     print(f"Validation Loss Epoch: {epoch_loss}")
#     print(f"Validation Accuracy Epoch: {epoch_accu}")

#     return epoch_accu


# for epoch in range(EPOCHS):
#     train(epoch)


# print('This is the validation section to print the accuracy and see how it performs')
# print('Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch')

# acc = valid(model, testing_loader)
# print("Accuracy on test data = %0.2f%%" % acc)


# os.makedirs('./models', exist_ok=True)

# # Saving the files for re-use

# output_model_file = './models/pytorch_distilbert_news.bin'
# output_vocab_file = './models/vocab_distilbert_news.bin'

# model_to_save = model
# torch.save(model_to_save, output_model_file)
# tokenizer.save_vocabulary(output_vocab_file)

# print('All files saved')
