#Another way
class MyNewNetwork(nn.Module):
    def __init__(self, num_classes):
        super(MyNewNetwork, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(3,8, kernel_size=5, padding=0),
                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(8,16, kernel_size=5, padding=0),
                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.classifier = nn.Sequential(nn.Linear(16*5*5, 64), nn.ReLU(),
                        nn.Linear(64, num_classes))
        
    def forward(self, input):
        input = self.conv1(input)
        input = self.conv2(input)
        input = input.reshape(input.size(0), -1) 
        input = self.classifier(input)

        return input

#More ways to define the model
class MyNewestNetwork(nn.Module):
    def __init__(self, num_classes):
        super(MyNewestNetwork, self).__init__()

        convmodel = []
        
        convmodel += [nn.Conv2d(3,8, kernel_size=5, padding=0),
                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]
        
        convmodel += [nn.Conv2d(8,16, kernel_size=5, padding=0),
                    nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)]

        self.convmodel = nn.Sequential(*convmodel)
        
        self.classifier = nn.Sequential(nn.Linear(16*5*5, 64), nn.ReLU(),
                        nn.Linear(64, num_classes))
        
    def forward(self, input):
        input = self.convmodel(input)
        input = input.reshape(input.size(0), -1) 
        input = self.classifier(input)

        return input

def train(nepochs, model, train_loader, test_loader, loss_fn, optimizer):
    statsrec = np.zeros((3,nepochs))

    for epoch in range(nepochs):  # loop over the dataset multiple times

        running_loss = 0.0
        n = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
                        
            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs.to(device)) # forward pass
            loss = loss_fn(outputs, labels.to(device)) # loss function
            loss.backward() # backward
            optimizer.step() #update params

            running_loss += loss.item()
            n += 1

        ltrn = running_loss/n
        ltst, atst = stats(test_loader, model)
        statsrec[:,epoch] = (ltrn, ltst, atst)
        print(f"epoch: {epoch} training loss: {ltrn: .3f}  test loss: {ltst: .3f} test accuracy: {atst: .1%}")
    
    return statsrec, model

#Example
nepochs = 10
model = MyNewNetwork(10).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

statistics, trainedmodel = train(nepochs, model, train_loader, test_loader, loss_fn, optimizer)
