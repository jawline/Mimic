import sample
import gan

import torch
import torch.autograd as autograd
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 100000
CRITIC_ITERS = 5
CUDA = False

# TODO: Figure out what this does
LAMBDA = .1

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def do_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(1, gan.DIM)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if CUDA else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if CUDA:
        interpolates = interpolates.cuda()

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda() if CUDA else torch.ones(disc_interpolates.size()),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train():

    # Our generator is trained to produce a valid input from noise, our discriminator is trained to detect a if an input is randomly generated
    generator = gan.Generator()
    discriminator = gan.Discriminator()

    generator.apply(weights_init)
    discriminator.apply(weights_init)

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
            real_data = torch.Tensor(real_data)
            if CUDA:
                real_data = real_data.cuda()
            real_data_v = autograd.Variable(real_data)

            discriminator.zero_grad()

            # train with real
            D_real = discriminator(real_data_v)
            D_real = D_real.mean()
            D_real.backward(neg_one)

            # train with fake
            noise = torch.randn( 1, gan.DIM)

            if CUDA:
                noise = noise.cuda()

            # TODO: Disable gradient
            noisev = autograd.Variable(noise)  # totally freeze netG
            fake = generator(noisev)
            fake = autograd.Variable(fake)
            D_fake = discriminator(fake)
            D_fake = D_fake.mean()
            D_fake.backward(one)
    
            # train with gradient penalty
            gradient_penalty = do_gradient_penalty(discriminator, real_data_v, fake)
            gradient_penalty.backward()
    
            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizer_d.step()
 
        ############################
        # (2) Update G network
        ###########################
        for p in discriminator.parameters():
            p.requires_grad = False  # to avoid computation
        generator.zero_grad()
    
        real_data = torch.Tensor(next(train_gen)['X'])

        if CUDA:
            real_data = real_data.cuda()

        real_data_v = autograd.Variable(real_data.data, requires_grad=True)
    
        noise = torch.randn(1, gan.DIM)

        if CUDA:
            noise = noise.cuda()

        noisev = autograd.Variable(noise.data)
        fake = generator(noisev)
        G = discriminator(fake)
        G = G.mean()
        G.backward(neg_one)
        G_cost = -G
        optimizer_g.step()
    
        # Write logs and save samples
        #lib.plot.plot('tmp/' + DATASET + '/' + 'disc cost', D_cost.cpu().data.numpy())
        #lib.plot.plot('tmp/' + DATASET + '/' + 'wasserstein distance', Wasserstein_D.cpu().data.numpy())
        #lib.plot.plot('tmp/' + DATASET + '/' + 'gen cost', G_cost.cpu().data.numpy())
        if (iteration + 1) % 100 == 0:
            parts = fake.reshape(sample.WINDOW_SIZE, sample.NUM_INP)
            for part in parts:
                print(part)

train()
