import torch
import torch.nn as nn


class AE(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_input_layer0 = torch.nn.Conv2d(3, 3, 3, padding=1, stride=8)
        self.encoder_input_layer1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.encoder_hidden_layer2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.encoder_hidden_layer3 = torch.nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.encoder_hidden_layer4 = torch.nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.decoder_hidden_layer0 = torch.nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.decoder_hidden_layer1 = torch.nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.decoder_hidden_layer2 = torch.nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
        self.decoder_output_layer = torch.nn.ConvTranspose2d(32, 3, 3, padding=1)
        self.decoder_output_layerto4096 = torch.nn.ConvTranspose2d(3, 3, 3, padding=1, stride=8)

        self.encoder = nn.Sequential(
            # self.encoder_input_layer0,
            self.encoder_input_layer1,
            self.encoder_hidden_layer2,
            self.encoder_hidden_layer3,
            self.encoder_hidden_layer4,
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            self.decoder_hidden_layer0,
            self.decoder_hidden_layer1,
            self.decoder_hidden_layer2,
            self.decoder_output_layer,
            nn.Sigmoid(),
        )

    def forward(self, features):
        '''
        activation=self.encoder_input_layer0(features)
        activation=self.encoder_input_layer1(activation)
        print(activation.shape)
        '''
        latent_space = self.encoder(features)
        sze_enc = latent_space.shape[-1]

        activation = self.decoder_hidden_layer0(latent_space, output_size=(sze_enc * 2, sze_enc * 2))
        activation = torch.nn.functional.leaky_relu(activation)
        activation = self.decoder_hidden_layer1(activation, output_size=(sze_enc * 4, sze_enc * 4))
        activation = torch.nn.functional.leaky_relu(activation)
        # print("d1",activation.shape)
        activation = self.decoder_hidden_layer2(activation, output_size=(sze_enc * 8, sze_enc * 8))
        activation = torch.nn.functional.leaky_relu(activation)
        # print("d2",activation.shape)
        activation = self.decoder_output_layer(activation)
        # activation = self.decoder_output_layerto4096(activation, output_size=(4096,4096))
        activation = torch.sigmoid(activation)

        # activation=self.decoder(latent_space)

        return activation, latent_space

        '''
        activation = self.decoder_hidden_layer0(activation,output_size=(sze_enc*2,sze_enc*2))
        activation = torch.nn.functional.leaky_relu(activation)
        activation = self.decoder_hidden_layer1(activation,output_size=(sze_enc*4,sze_enc*4))
        activation = torch.nn.functional.leaky_relu(activation)
        #print("d1",activation.shape)
        activation = self.decoder_hidden_layer2(activation,output_size=(sze_enc*8,sze_enc*8))
        activation = torch.nn.functional.leaky_relu(activation)
        #print("d2",activation.shape)
        activation = self.decoder_output_layer(activation)
        #activation = torch.sigmoid(activation)
        #print("d3",activation.shape)
        return activation
        '''
