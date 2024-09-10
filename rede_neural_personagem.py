import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #origem do 12:
        # fome e vida = 2 entradas
        # direcao da visao = 1 entrada
        # ultima posicao (x e y) = 2 entradas
        # x e y comida mais proxima = 2 entradas
        # x e y lobsomem = 2 entradas
        # x e y dos 3 humanos mais proximos = 6 entradas
        self.fc1 = nn.Linear(11, 10)#quero trocar para 12 depois
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 3)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
def salvar_pesos_binario(model, filename='pesos.pth'):
    torch.save(model.state_dict(), filename)
    print(f"Pesos salvos em {filename}")

def carregar_pesos_binario(model, filename='pesos.pth'):
    model.load_state_dict(torch.load(filename))
    print(f"Pesos carregados de {filename}")
