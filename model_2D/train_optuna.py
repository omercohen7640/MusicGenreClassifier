import model_2D.resnet_dropout as resnet_dropout
import model_2D.datamanager_ver2 as datamanager_ver2
from model_2D.train_2D import *

def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    music_classify= resnet_dropout.resnet18(num_classes = 10)
    music_classify= music_classify.to(device)
    print("loading data...")
    train_loader,valid_loader,_,_ = datamanager_ver2.get_dataloader(hparams, dataset_size=hparams.optuna_ds)
    test_acc = train_for_optuna(trial=trial,music_classify=music_classify,train_loader=train_loader,valid_loader=valid_loader)
    return test_acc


def train_for_optuna(trial,music_classify, train_loader, valid_loader, test_loader=None):
    now = dt.now()
    dt_string = now.strftime("%d%m%Y%H%M")
    path = os.path.join(os.getcwd(),"trial_"+dt_string)
    os.mkdir(path)


    print("starting training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    warmup_factor = trial.suggest_float("warmup_factor", 1e1, 1e3, log=True)
    init_lr = lr/warmup_factor
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    optimizer = getattr(torch.optim, optimizer_name)(music_classify.parameters(), lr=init_lr, weight_decay=hparams.weight_decay)
    sched_factor = trial.suggest_float("sched_factor",0,1)
    patience = trial.suggest_int("patience",1,6)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=sched_factor, patience=patience,
                                               verbose=True)
    #warmup section:
    warmup_epochs = trial.suggest_int("warmup_ephocs",2,5)
    num_steps = warmup_epochs*len(train_loader)

    warmuper = my_warmup.LinearWarmuper(optimizer=optimizer,steps=num_steps, factor = warmup_factor)
    print("warmup section...")
    for w_epoch in range(warmup_epochs+1):
        print("info: lr={}".format(get_lr(optimizer)))
        total_tracks = 0
        total_correct = 0
        epoch_time = time.time()
        for i, data in enumerate(train_loader):
            running_loss = 0.0
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            running_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            warmuper.step()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| time: {}".format(w_epoch, running_loss,
                                                                          model_accuracy,epoch_time)
        print(log)
    best_valid_acc = 0
    # warmup_scheduler = warmup.UntunedLinearWarmup(optimizer, last_step=5)
    ephocs = trial.suggest_int("ephocs",10,30)
    print("starting train loop...")
    for epoch in range(1, ephocs + 1):
        print("info: lr={}".format(get_lr(optimizer)))
        music_classify.train()
        running_loss = 0.0
        epoch_time = time.time()
        total_tracks = 0
        total_correct = 0

        for i, data in enumerate(train_loader):
            waveform, label = data
            waveform = waveform.to(device)
            label = label.to(device)
            size = waveform.size()
            inputs = torch.zeros(size[0], 3, size[1], size[2])
            inputs[:, 0, :, :] = waveform
            inputs[:, 1, :, :] = waveform
            inputs[:, 2, :, :] = waveform
            inputs = inputs.to(device)
            outputs = music_classify(inputs)
            loss = criterion(outputs, label)
            _, predicted = torch.max(outputs.data, 1)
            total_tracks += label.size(0)
            total_correct += (predicted == label).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item()
        running_loss /= len(train_loader)
        model_accuracy = total_correct / total_tracks * 100
        epoch_time = time.time() - epoch_time
        valid_accuracy, _, valid_loss = calculate_accuracy(music_classify, valid_loader, device, criterion)
        log = "Epoch: {}  training loss: {:.3f} | train acc: {}| valid acc: {} | time: {}".format(epoch, running_loss,
                                                                                                  model_accuracy,
                                                                                                  valid_accuracy,
                                                                                                  epoch_time)
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            path_save = os.path.join(path,"best_model")
            torch.save(music_classify.state_dict(),path_save)

        print(log)
        scheduler.step(metrics=valid_accuracy)
        # warmup_scheduler.dampen()
    path_save = os.path.join(path, "last_model")
    torch.save(music_classify.state_dict(), path_save)
    return valid_accuracy

def run_parameter_tuning():
    logger = logging.getLogger()

    logger.setLevel(logging.INFO)  # Setup the root logger.
    logger.addHandler(logging.FileHandler("../foo.log", mode="w"))

    optuna.logging.enable_propagation()  # Propagate logs to the root logger
    study = optuna.create_study(study_name= "music classifier", direction="maximize",sampler=optuna.samplers.TPESampler())
    logger.info("Start optimization.")
    study.optimize(objective, n_trials=hparams.number_of_trials)
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    optuna.visualization.plot_param_importances(study)
    joblib.dump(study, "study.pkl")