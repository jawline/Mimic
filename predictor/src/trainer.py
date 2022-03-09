from datetime import datetime

import torch
from torch.cuda.amp import autocast
from torch import nn

ROUND_SZ = 10000

def train(data_loader, load_fn, path, device):

    cpu = torch.device('cpu')
    command_generator, optimizer, scheduler = load_fn(path, device)
    criterion = nn.CrossEntropyLoss()
    running_loss = torch.zeros(1, device=device)
    data_loader = iter(data_loader)

    def step():

        print("Starting batch")
        running_loss.zero_()

        for i in range(ROUND_SZ):

            if i % (ROUND_SZ / 10) == 0:
                print("Batch completion:", (float(i) / float(ROUND_SZ)) * 100., "%")

            seq = next(data_loader).long().to(device)
            inputs = seq[:,:-1]
            labels = seq[:,1:]

            optimizer.zero_grad()

            with autocast():
                outputs = command_generator(inputs, command_generator.get_tgt_mask(inputs.size(1)).to(device))
                #print("Shapes: ", outputs.view(-1, 256).shape, labels.reshape(-1).shape)
                loss = criterion(outputs.view(-1, 256), labels.reshape(-1))

            loss.backward()
            optimizer.step()
            running_loss.add_(loss.detach())

            seq = seq.detach().to(cpu)
            del inputs
            del labels
            del seq

        result = running_loss / ROUND_SZ
        return result

    def save(name):
        torch.save(command_generator.state_dict(), "./" + name + ".model")
        torch.save(optimizer.state_dict(), "./" + name + ".optimizer")

    current_iteration = 0

    while True:
        loss = step()
        scheduler.step(loss)

        print("Loss:", loss.item())
        print("LR:", optimizer.param_groups[0]['lr'])

        print("Saving checkpoint")

        # Timestamp every 10th epoch to test fits later
        if current_iteration % 10 == 0:
            save(str(int(datetime.now().timestamp())))

        save("./last.checkpoint")
        print("Saved checkpoint")

        current_iteration += 1

    return command_generator.eval()
