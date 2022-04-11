from datetime import datetime

import torch
from torch.cuda.amp import autocast
from torch import nn

from sample import MAX_WINDOW_SIZE

ROUND_SZ = 1000

def train(data_loader, validation_loader, load_fn, model_dir, load_path, device):

    cpu = torch.device('cpu')

    print("Train called with: ", model_dir, load_path)

    # Send none to load_fn if load_path is None otherwise append the model dir to it
    path = None

    if load_path != None:
        path = model_dir + load_path

    command_generator, optimizer, scheduler = load_fn(path, device)
    criterion = nn.CrossEntropyLoss()
    running_loss = torch.zeros(1, device=device)
    data_loader = iter(data_loader)
    validation_loader = iter(validation_loader)

    def step(ldr, sz, backprop):

        print("Starting batch")
        running_loss.zero_()

        for i in range(sz):

            if i % (sz / 10) == 0:
                print("Batch completion:", (float(i) / float(sz)) * 100., "%")

            seq = next(ldr).to(device)
            inputs = seq[:,:-1]
            labels = seq[:,1:]

            optimizer.zero_grad()
            #torch.autograd.set_detect_anomaly(True)
            #outputs = command_generator(inputs, command_generator.get_tgt_mask(inputs.size(1)).to(device))
            #print("Shapes: ", outputs.view(-1, 256).shape, labels.reshape(-1).shape)
            logits = command_generator(inputs) 
            #logits = logits.reshape(-1, logits.shape[-1])
            #labels = labels.reshape(-1)
            #print(logits.shape, labels.shape)
            loss = criterion(logits, labels)
            #print("Loss:", loss)
            #loss = criterion(outputs.view(-1, 256), labels.reshape(-1))

            if backprop:
                loss.backward()
                optimizer.step()
            running_loss.add_(loss.detach())

            seq = seq.detach().to(cpu)
            del inputs
            del labels
            del seq

        result = running_loss / sz
        return result

    def save(name):
        print("Saving model to path: " + name)
        torch.save(command_generator.state_dict(), "./" + name + ".model")
        torch.save(optimizer.state_dict(), "./" + name + ".optimizer")

    epoch = 1
   
    tolerence_validation_base = 4
    tolerence_validation = tolerence_validation_base
    last_validation = None

    while True:

        print("Pre-step LR:", optimizer.param_groups[0]['lr'])

        # Do a ROUND_SZ of training and backprop
        loss = step(data_loader, ROUND_SZ, True)

        # Feed the current epoch and loss (1-indexed not 0-indexed) into our scheduler function to adjust the LR
        scheduler.step(loss)

        # Do a round 10 of validation with no backprop
        validation_loss = step(validation_loader, 200, False)

        print("Loss:", loss.item())
        print("Validation loss:", validation_loss.item())
        print("LR:", optimizer.param_groups[0]['lr'])

        print("Saving checkpoint")

        # Timestamp every 10th epoch to test fits later
        if epoch % 3 == 0:
            save(model_dir + "/" + str(int(datetime.now().timestamp())))

        save(model_dir + "/last.checkpoint")

        if last_validation != None:

            if validation_loss > last_validation:
                print("Decremented tolerence because validation loss went down")
                tolerence_validation = tolerence_validation - 1
            else:
                tolerence_validation = tolerence_validation_base

            #if tolerence_validation <= 0:
            #    sys.exit("Early exit because validation loss has stopped going down")
        last_validation = validation_loss

        print("Saved checkpoint")

        epoch += 1

    return command_generator.eval()
