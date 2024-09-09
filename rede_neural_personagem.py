import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #origem do 12:
        # fome e vida = 2 entradas
        # x e y comida mais proxima = 2 entradas
        # x e y lobsomem = 2 entradas
        # x e y dos 3 humanos mais proximos = 6 entradas
        self.fc1 = nn.Linear(8, 10)#quero trocar para 12 depois
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
    

def treinar(model, optimizer, criterio, epochs=100):
    for epoch in range(epochs):
        # Exemplo de treinamento (com entradas e saídas fictícias)
        entradas = torch.randn(10, 12)  # Lote de 10 entradas com 12 características cada
        saidas_verdadeiras = torch.randn(10, 3)  # Lote de 10 saídas reais com 3 valores

        # Passa as entradas pela rede neural
        saidas_preditas = model(entradas)
        
        # Calcula o erro
        loss = criterio(saidas_preditas, saidas_verdadeiras)
        
        # Backpropagation e ajuste dos pesos
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
