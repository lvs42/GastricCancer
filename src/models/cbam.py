import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    """ Decide para cada imagen como de importante es la informacion de cada canal"""

    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)# Pooling de la media


        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = int(num_channels / reduction_ratio)



        self.mlp = nn.Sequential(
            nn.Linear(num_channels, reduced_channels), #num_channels -> reduced_channels
            nn.ReLU(),
            nn.Linear(reduced_channels, num_channels) # volvemos a num_channels
        )

        # Normalizamos
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        x tiene forma:
        (batch, C, H, W)

        Devuelve:
        (batch, C, 1, 1)
        """

        # Aplica el pooling de media sobre x.

        avg_features = self.avg_pool(x)

        avg_features = torch.flatten( avg_features, start_dim = 1)

    
        avg_attention = self.mlp(avg_features) # Pasa las caracteristicas por la mlp



        max_features = self.max_pool(x)
        max_features = torch.flatten( max_features, start_dim = 1)
        max_attention = self.mlp(max_features)



        # Suma la atencion obtenido por las dos opciones
        attention = avg_attention + max_attention


        attention = self.sigmoid(attention) #valores entre 0 y 1



        # Recuperamos batch y numero de canales
        batch_size = attention.shape[0]
        num_channels = attention.shape[1]


        attention = attention.view(batch_size, num_channels, 1, 1) #vovlem,so a (batch, C, 1, 1)
        

        return attention



class SpatialAttention(nn.Module):
    """ Atencion espacial será el encargado de decidir qué posiciones espaciales del mapa tiene las características más improtantes"""

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = kernel_size // 2


        self.conv = nn.Conv2d(
            in_channels = 2,
            out_channels = 1,
            kernel_size= kernel_size,
            padding = padding
        )


        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        """
        x tiene forma:
        (batch, C, H, W)

        Devuelve:
        (batch, 1, H, W)
        """


        avg_map = torch.mean(x, dim = 1, keepdim = True)


        max_result = torch.max(x,  dim = 1,  keepdim = True) #devuelve indices y values

        max_map = max_result.values


        combined = torch.cat((avg_map, max_map),  dim=1)

        # la atencion son los dos mapas concatenados
        attention = self.conv(combined)

        attention = self.sigmoid(attention)


        return attention



class CBAM(nn.Module):
    """ Modulo CBAM completo"""

    def __init__(self,  num_channels,  reduction_ratio=16,  spatial_kernel_size=7):
        super().__init__()


        self.channel_attention = ChannelAttention( num_channels, reduction_ratio) #self se pasa automaticamente

        self.spatial_attention = SpatialAttention(spatial_kernel_size)


    def forward(self, x):
        """
        x:
        (batch, C, H, W)

        salida:
        (batch, C, H, W)
        """

        channel_weights = self.channel_attention(x)


        x_channel = x * channel_weights

        spatial_weights = self.spatial_attention(x_channel)


        x_final = x_channel * spatial_weights


        return x_final


"""
if __name__ == "__main__":

    modulo = CBAM(
        num_channels=1280
    )

    x = torch.randn(
        4,
        1280,
        7,
        7
    )

    salida = modulo(x)

    print("Forma entrada:")
    print(x.shape)

    print("Forma salida:")
    print(salida.shape)

    assert salida.shape == x.shape

    assert not torch.equal(
        salida,
        x
    )

    print("CBAM funciona correctamente.")
    """