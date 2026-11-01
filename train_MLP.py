import torch
import torch.nn as nn
import split_train 

class MLP(nn.Module):
    def __init__(self,input_size, hidden_size=128):
        super().__init__()

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

#forward pass
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    

epochs = 20
learning_rate = 0.001

X_train, X_dev, y_train, y_dev = split_train("optdigits.tra")

opti_model = MLP(input_size=64)
pen_model = MLP(input_size=16)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)

#loss and optimizer
L = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    opti_model.parameters(),
    lr= learning_rate
)

for epoch in range(epochs):
    opti_model.train()
    optimizer.zero_grad()

    outputs = opti_model(X_train_t) #calls forward
    loss = L(outputs, y_train_t)
    loss.backward()
    optimizer.step()
    



    