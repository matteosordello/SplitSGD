# Meaning of the parameters:
# net -> neural network that we want to train
# K -> number of single threads we use
# t1 -> length of first single thread
# q -> proportion of negative gradient coherences to declare stationarity
# w -> number of windows used in the diagnostic (the total length of the diagnostic is fixed at 1 epoch)
# gamma -> discount factor for learning rate
# lr -> initial laerning rate
# mom -> momentum

def SplitSGD(net, K, t1, q, w, gamma, lr, mom):

    criterion_splitsgd = nn.CrossEntropyLoss()
    optimizer_splitsgd = optim.SGD(net.parameters(), lr=lr, momentum=mom)
    real_epoch = 0
    l = len(train_loader)
    accuracy_splitsgd = []


    for k in range(K):
        for t in range(t1):
            for images, labels in train_loader:
            ######### add this line if used on Mnist
                images = images.view(-1, 28*28).requires_grad_()
            #########
                optimizer_splitsgd.zero_grad()
                outputs = net(images)
                loss = criterion_splitsgd(outputs, labels)
                loss.backward()
                optimizer_splitsgd.step()

            real_epoch += 1 
            correct = 0
            total = 0
            for images, labels in test_loader:
            ######### add this line if used on Mnist
                images = images.view(-1, 28*28).requires_grad_()
            #########
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100*correct/total
            accuracy_splitsgd.append(accuracy)

            print('epoch: %d, test accuracy: %.2f' % (real_epoch, accuracy))        



        net1 = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)                                                                 
        net1.load_state_dict(net.state_dict())                                        
        optimizer1 = torch.optim.SGD(net1.parameters(), lr = lr, momentum = mom)     
        optimizer1.load_state_dict(optimizer_splitsgd.state_dict())                         
        criterion1 = nn.CrossEntropyLoss()

        net2 = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)                                                            
        net2.load_state_dict(net.state_dict())                                       
        optimizer2 = torch.optim.SGD(net2.parameters(), lr = lr, momentum = mom)     
        optimizer2.load_state_dict(optimizer_splitsgd.state_dict())                          
        criterion2 = nn.CrossEntropyLoss()        

        # Copy the two net so we can get back the parameters    
        init_params1 = copy.deepcopy(net1)
        init_params2 = copy.deepcopy(net2)

        dot_prod = []
        for i, (images, labels) in enumerate(train_loader):
        ######### add this line if used on Mnist
            images = images.view(-1, 28*28).requires_grad_()
        #########
            if i%2 == 0:
                optimizer1.zero_grad()
                outputs = net1(images)
                loss = criterion1(outputs, labels)
                loss.backward()
                optimizer1.step()
            if i%2 == 1:
                optimizer2.zero_grad()
                outputs = net2(images)
                loss = criterion2(outputs, labels)
                loss.backward()
                optimizer2.step()
            if i%int(l/w) == int(l/w)-1:                  # We are using w windows dor each thread of the diagnostic
                fin_params1 = net1.state_dict()
                fin_params2 = net2.state_dict()

                for param_tensor in dict(net1.named_parameters()).keys():
                    p1 = fin_params1[param_tensor] - init_params1.state_dict()[param_tensor]
                    p2 = fin_params2[param_tensor] - init_params2.state_dict()[param_tensor]
                    dot_prod.append(torch.sum(p1*p2))

                init_params1 = copy.deepcopy(net1)
                init_params2 = copy.deepcopy(net2)


        stationarity = (sum([dot_prod[i] < 0 for i in range(len(dot_prod))]) >= q*len(dot_prod))
        if stationarity:
            lr = lr*gamma
            # here you can also let the next single thread be longer by a factor gamma
            # by adding -> t1 = int(t1/gamma)

        net = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)  
        beta = 0.5 
        params1 = net1.state_dict()
        params2 = net2.state_dict()
        for name1 in params1.keys():
            if name1 in params2.keys():
                params2[name1].data.copy_(beta*params1[name1].data + (1-beta)*params2[name1].data)

        net.load_state_dict(params2)
        optimizer_splitsgd = torch.optim.SGD(net.parameters(), lr = lr, momentum = mom)
        criterion_splitsgd = nn.CrossEntropyLoss()


        real_epoch += 1
        correct = 0
        total = 0
        for images, labels in test_loader:
        ######### add this line if used on Mnist
            images = images.view(-1, 28*28).requires_grad_()
        #########
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100*correct/total
        accuracy_splitsgd.append(accuracy)

        print('D -> epoch: %d, test accuracy: %.2f, stationarity: %s, negative dot products: %i out of %i, 
        learning rate: %.4f' % (real_epoch, accuracy, bool(stationarity), 
                 sum([dot_prod[i] < 0 for i in range(len(dot_prod))]), len(dot_prod), lr))
        
    return(accuracy_splitsgd)


