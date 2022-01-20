import sample
import gan

import torch
import torch.autograd as autograd
import torch.optim as optim

EPOCHS = 100000
CRITIC_ITERS = 5
CUDA = False

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train():

    # Our generator is trained to produce a valid input from noise, our discriminator is trained to detect a if an input is randomly generated
    generator = gan.Generator()
    discriminator = gan.Discriminator()

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    #print(generator)
    #print(discriminator)

    if CUDA:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    one = torch.tensor(1, dtype=torch.float)
    neg_one = one * -1
    if CUDA:
        one = one.cuda()
        neg_one = neg_one.cuda()

    print("RR")

    train_gen, valid_data, test_data  = sample.create_data_split(sample.training_files("./training_data/"), 0.1, 0.3, 1, 1, 1)

    for iteration in range(EPOCHS):
        
        ############################
        # (1) Update D network
        ###########################
        for p in discriminator.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(CRITIC_ITERS):

            # TODO: Not sure if we should use train-gen here
            real_data = next(train_gen)['X']
            print(real_data)
            real_data = torch.Tensor(real_data)
            if CUDA:
                real_data = real_data.cuda()
            real_data_v = autograd.Variable(real_data)

            discriminator.zero_grad()

            # train with real
            D_real = discriminator(real_data_v)
            print("D-real", D_real)
            D_real = D_real.mean()
            D_real.backward(neg_one)

            # train with fake
            noise = torch.randn(gan.DIM, gan.DIM)

            if CUDA:
                noise = noise.cuda()

            noisev = autograd.Variable(noise)  # totally freeze netG
            fake = autograd.Variable(generator(noisev, real_data_v).data)
            inputv = fake
            D_fake = discriminator(inputv)
            D_fake = D_fake.mean()
            D_fake.backward(one)
    
            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data_v.data, fake.data)
            gradient_penalty.backward()
    
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()
 
          ############################
          # (2) Update G network
          ###########################
        for p in discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
    
        _data = data.next()
        real_data = torch.Tensor(_data)
        if use_cuda:
            real_data = real_data.cuda()
        real_data_v = autograd.Variable(real_data)
    
        noise = torch.randn(BATCH_SIZE, 2)
        if use_cuda:
            noise = noise.cuda()
        noisev = autograd.Variable(noise)
        fake = netG(noisev, real_data_v)
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()
    
        # Write logs and save samples
        lib.plot.plot('tmp/' + DATASET + '/' + 'disc cost', D_cost.cpu().data.numpy())
        lib.plot.plot('tmp/' + DATASET + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
        lib.plot.plot('tmp/' + DATASET + '/' + 'gen cost', G_cost.cpu().data.numpy())
        if iteration % 100 == 99:
            lib.plot.flush()
            generate_image(_data)
        lib.plot.tick()

train()
