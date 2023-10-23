class Train_Test():

    def __init__(self,epochs:int = epochs):
        self.EPOCHS = 5
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.001)
    def train(self):
        for epoch in range(self.EPOCHS):
            print(f"Epoch: {epoch}\n------")
            train_loss = 0
            for batch, (X, y_tuple) in (enumerate(train_dataloader)):
                model.train()
                # if len(y_tuple)<BATCH_SIZE:
                #     continue
                y_pred = torch.tensor(model(X.unsqueeze(1)))
                batch_loss = 0
                y = np.asarray(y_tuple)
                for i in range(len(y)):
                    word_loss = 0
                    try:
                        if len(y[i])<len(y_pred[i]):
                            # for j in range(len(y_pred[i])-len(y[i])):
                            while len(y_pred[i]) != len(y[i]):
                                # print(j)
                                y[i]+='-'
                        # print(len(y_pred[i]),y[i])
                        for j in range(len(y_pred[i])):
                            if y[i][j]=='-' or y[i][j] ==' ':
                                continue
                            loss = self.loss_fn(y_pred[i][j][:],torch.tensor(LABELS[y[i][j]]))
                            loss.requires_grad = True
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                            word_loss+=loss
                    except:
                        print("error")
                        pass
                    word_loss/=len(y_pred[i])
                    batch_loss+=word_loss

                batch_loss/=len(y)   
                train_loss+=batch_loss  

                if batch % 10 == 0:
                    print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
                    print(batch_loss) 
                    
            train_loss /= len(train_dataloader) 
            print(train_loss)       

    def test(self):
        model.eval()
        test_loss = 0
        
        with torch.inference_mode():
        # Loop through DataLoader batches
            for batch, (X, y_tuple) in (enumerate(test_dataloader)): 
                if len(y_tuple)<BATCH_SIZE:
                    continue
                # print(X.unsqueeze(1).shape)
                y_pred = torch.tensor(model(X.unsqueeze(1)))
                batch_loss = 0
                y = np.asarray(y_tuple)
                for i in range(len(y)):
                    word_loss = 0
                    try:
                        if len(y[i])<len(y_pred[i]):
                            for j in range(len(y_pred[i])-len(y[i])):
                                y[i]+='-'
                        # print(len(y_pred[i]),y[i])
                        for j in range(len(y_pred[i])):
                            if y[i][j]=='-' or y[i][j] ==' ':
                                continue
                            loss = self.loss_fn(y_pred[i][j][:],torch.tensor(LABELS[y[i][j]]))
                            loss.requires_grad = True
                            word_loss+=loss
                    except:
                        pass
                    word_loss/=len(y_pred[i])
                    batch_loss+=word_loss

                batch_loss/=len(y)   
                test_loss+=batch_loss  

                if batch % 10 == 0:
                    print(f"Looked at {batch * len(X)}/{len(test_dataloader.dataset)} samples.")
                    print(batch_loss) 
                    
            test_loss /= len(test_dataloader) 
            print(test_loss)
            return test_loss