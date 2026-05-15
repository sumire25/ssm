import os
import torch
import torch.nn as nn
import torchvision
import torch.optim
import argparse
import image_data_loader
import model
from torch.optim.lr_scheduler import CosineAnnealingLR


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(args):
    # Asegurar que los directorios de salida existan
    os.makedirs("trained_weights", exist_ok=True)
    os.makedirs("training_data_captures", exist_ok=True)

    # Instanciar el modelo con Mamba (SSM)
    lfd_net = model.LFD_Net_SSM().cuda()
    lfd_net.apply(weights_init)

    # Cargar conjuntos de datos de forma independiente
    training_data = image_data_loader.hazy_data_loader(args["train_original"], args["train_hazy"])
    validation_data = image_data_loader.hazy_data_loader(args["val_original"], args["val_hazy"], mode="val")
    
    training_data_loader = torch.utils.data.DataLoader(training_data, batch_size=8, shuffle=True, num_workers=2,
                                                       pin_memory=True)
    validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=8, shuffle=True, num_workers=2,
                                                         pin_memory=True)

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(lfd_net.parameters(), lr=float(args["learning_rate"]), weight_decay=0.0001)
    
    num_of_epochs = int(args["epochs"])
    
    # Corrección Matemática del Scheduler: 
    # Como scheduler.step() se llama por iteración, T_max debe ser el número total de iteraciones.
    total_iterations = num_of_epochs * len(training_data_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iterations)
    
    lfd_net.train()

    for epoch in range(num_of_epochs):
        for iteration, (hazefree_image, hazy_image) in enumerate(training_data_loader):
            hazefree_image = hazefree_image.cuda()
            hazy_image = hazy_image.cuda()
            dehaze_image = lfd_net(hazy_image)
            
            loss = criterion(dehaze_image, hazefree_image)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lfd_net.parameters(), 0.1)
            
            optimizer.step()
            scheduler.step() # El descenso del coseno ocurre paso a paso
            
            if ((iteration + 1) % 10) == 0:
                print("Epoch:", epoch, "| Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % 200) == 0:
                torch.save(lfd_net.state_dict(), "trained_weights/" + "Epoch_" + str(epoch) + '.pth')

        # Fase de Validación (Asegurar que los gradientes no se calculen ni almacenen)
        lfd_net.eval()
        with torch.no_grad():
            for iter_val, (hazefree_image, hazy_image) in enumerate(validation_data_loader):
                hazefree_image = hazefree_image.cuda()
                hazy_image = hazy_image.cuda()
                dehaze_image = lfd_net(hazy_image)
                torchvision.utils.save_image(torch.cat((hazy_image, dehaze_image, hazefree_image), 0),
                                             "training_data_captures/" + str(iter_val + 1) + ".jpg")
        lfd_net.train()

        # Guardar pesos al final de cada época
        torch.save(lfd_net.state_dict(), "trained_weights/" + "Epoch_" + str(epoch) + '.pth')


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    # Argumentos de Entrenamiento
    ap.add_argument("-th", "--train_hazy", required=True, help="path to hazy training images")
    ap.add_argument("-to", "--train_original", required=True, help="path to original training images")
    
    # Argumentos de Validación
    ap.add_argument("-vh", "--val_hazy", required=True, help="path to hazy validation images")
    ap.add_argument("-vo", "--val_original", required=True, help="path to original validation images")
    
    # Hiperparámetros
    ap.add_argument("-e", "--epochs", required=True, help="number of epochs for training")
    ap.add_argument("-lr", "--learning_rate", required=True, help="learning rate for training")

    args = vars(ap.parse_args())

    train(args)