from model_2D.train_2D import *


def calculate_accuracy_1d(model, dataloader, device, criterion):
    model = model.to(device)
    model.eval()  # put in evaluation mode
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    runningn_loss = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            runningn_loss += loss
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1
    runningn_loss /= len(dataloader)
    model_accuracy = total_correct / total_images * 100
    return model_accuracy, confusion_matrix, runningn_loss


def train_1d(train_loader, valid_loader, test_loader, music_classify):
    print("starting training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    music_classify = music_classify.to(device)
    now = dt.now()
    lr_iterarion = []
    vall_acc_epoch = []
    train_loss_iter = []
    train_acc_epoch = []
    dt_string = now.strftime("%d%m%Y%H%M")
    path = os.path.join(os.getcwd(), "trial_" + dt_string)
    os.mkdir(path)
    criterion = nn.CrossEntropyLoss()
    lr = hparams.learning_rate
    warmup_factor = hparams.warmup_factor
    init_lr = lr / warmup_factor

    optimizer = torch.optim.Adam(music_classify.parameters(), lr=init_lr, weight_decay=hparams.weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, threshold=hparams.threshold, mode='max',
                                               factor=hparams.factor, patience=hparams.patience, verbose=True)
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    print("begin warmup loop...")
    warmup_epochs = hparams.warmup_epochs
    num_steps = warmup_epochs * len(train_loader)
    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer, steps=num_steps, factor=1e2)
    for w_epoch in range(warmup_epochs + 1):
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        print("info: lr={}".format(get_lr(optimizer)))
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            outputs = music_classify(waveform)
            loss = criterion(outputs, label)
            train_loss_iter.append(loss)
            running_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()
            lr_iterarion.append(get_lr(optimizer))
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| time: {}".format(w_epoch, running_loss, model_accuracy,
                                                                                  epoch_time)
        print(log)
    best_valid_acc = 0
    number_of_ephocs = hparams.num_epochs
    print("starting train loop...")
    for epoch in range(1, number_of_ephocs + 1):
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0
        print("info: lr={}".format(get_lr(optimizer)))

        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            outputs = music_classify(waveform)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_iterarion.append(get_lr(optimizer))
            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy, _, valid_loss = calculate_accuracy_1d(music_classify, valid_loader, device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss,
                                                                                                  model_accuracy,
                                                                                                  valid_accuracy,
                                                                                                  epoch_time)
        train_acc_epoch.append(model_accuracy)
        vall_acc_epoch.append(valid_accuracy)
        print(log)
        scheduler.step(metrics=valid_accuracy)
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            path_save = os.path.join(path, "best_model")
            torch.save(music_classify.state_dict(), path_save)
    lr_iterarion = np.array(lr_iterarion)
    vall_acc_epoch = np.array(vall_acc_epoch)
    train_loss_iter = np.array(train_loss_iter)
    train_acc_epoch = np.array(train_acc_epoch)
    np.save(os.path.join(path, 'lr_iteration'), lr_iterarion)
    np.save(os.path.join(path, 'vall_acc_epoch'), vall_acc_epoch)
    np.save(os.path.join(path, 'train_loss_iter'), train_loss_iter)
    np.save(os.path.join(path, 'train_acc_epoch'), train_acc_epoch)
    if test_loader is not None:
        test_accuracy, _, _ = calculate_accuracy_1d(music_classify, test_loader, device, criterion=criterion)
        path_save = os.path.join(path, "last_model")
        torch.save(music_classify.state_dict(), path_save)
        return test_accuracy

if __name__ == '__main__':
    model1dver1 = model2.Music1DCNN_ver2()
    train_loader, valid_loader, test_loader = DataManager.get_dataloader(hparams)
    train_1d(train_loader,valid_loader,test_loader,model1dver1)