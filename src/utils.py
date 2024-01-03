import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd




class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        }

    def __len__(self):
        return self.len
    
class DistillBERTClass(torch.nn.Module):
    def __init__(self, base_model, ff_layers=3):
        super(DistillBERTClass, self).__init__()
        self.l1 = base_model # DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 1024)
        self.post_classifier = torch.nn.Linear(1024, 1024)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(1024, 4)
        self.ff_layers = ff_layers

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        for i in range(self.ff_layers):
            pooler = self.post_classifier(pooler)
            pooler = torch.nn.ReLU()(pooler)
            pooler = self.dropout(pooler)

        output = self.classifier(pooler)
        return output
    
# Function to calcuate the accuracy of the model

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct


encode_dict = {}

def encode_cat(x):
    global encode_dict

    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]
    
def get_train_test_data(data_path= '/home/patidarritesh/smartsense/Car_category_classification/data/data_sheet.csv'):
    # Import the csv into pandas dataframe and add the headers
    # df = pd.read_csv('/home/patidarritesh/smartsense/newsCorpora.csv', sep='\t', names=['ID','TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
    df = pd.read_csv(data_path)
    print("Total records: ", len(df))
    df = df.dropna() 
    print("Total records after removing null values: ", len(df))

    # rename the columns text to TITLE and folder_name to CATEGORY
    df.rename(columns={'text':'TITLE', 'folder_name':'CATEGORY'}, inplace=True)
    print("Unique categories: ", df.CATEGORY.unique())


    df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

    print("Unique categories after encoding: ", df.ENCODE_CAT.unique())

    global encode_dict
    print("Encode dict: ", encode_dict)


    # Creating the dataset and dataloader for the neural network

    train_size = 0.8
    train_dataset=df.sample(frac=train_size,random_state=200)
    test_dataset=df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))


    return train_dataset, test_dataset