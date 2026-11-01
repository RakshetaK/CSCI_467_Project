import torch
import torch.nn as nn
from split_train import split_train

class MLP(nn.Module):
    def __init__(self,input_size, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

#forward pass
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    

epochs = [200, 300, 400, 500]
learning_rate = [0.001, 0.0005]

X_train, X_dev, y_train, y_dev = split_train("C:/Users/15rak/Downloads/CSCI_467_Project/dataset_467/pen+based+recognition+of+handwritten+digits/pendigits.tra")


pen_model = MLP(input_size=16)

X_train_t = torch.tensor(X_train / 100.0, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)


torch.manual_seed(42)

#loss and optimizer
L = nn.CrossEntropyLoss()
for learning in learning_rate:
    for e in epochs:
        torch.manual_seed(42)
        opti_model = MLP(input_size=16)
        optimizer = torch.optim.Adam(
            opti_model.parameters(),
            lr= learning
        )
        for epoch in range(e):
            opti_model.train()
            optimizer.zero_grad()

            outputs = opti_model(X_train_t) #calls forward
            loss = L(outputs, y_train_t)
            loss.backward()
            optimizer.step()

            # print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


        opti_model.eval()

        with torch.no_grad():
            outputs = opti_model(X_train_t)

            predictions = torch.argmax(outputs, dim=1)

            accuracy = (predictions == y_train_t).float().mean()
        
            print(f"Training accuracy for learning rate {learning} and epoch {e}: {accuracy.item()*100:.2f}%")
        


    #dev set
        X_dev_t = torch.tensor(X_dev / 100.0, dtype=torch.float32)
        y_dev_t = torch.tensor(y_dev, dtype = torch.long)

        opti_model.eval()

        with torch.no_grad():
            outputs = opti_model(X_dev_t)
            predictions = torch.argmax(outputs, dim=1)

            dev_accuracy = (predictions == y_dev_t).float().mean()

        print(f"Dev accuracy: {dev_accuracy.item()*100:.2f}%")


    #best run was 500 epochs 0.001 lr