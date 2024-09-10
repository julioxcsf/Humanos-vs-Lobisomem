'''
    Autor: Julio Cesar S. Fernandes
    09/2024

'''


import pygame
import math
from janela import Janela
from personagem import Humano,Lobisomem,Comida

import torch
import torch.nn as nn
import torch.optim as optim
import rede_neural_personagem

index = 0

def treinar_personagem(personagem):
    optimizer = optim.Adam(personagem.cerebro.parameters(), lr=0.001)
    criterio = nn.MSELoss()

    # Verifica se é um humano ou lobisomem para usar o método correto
    if isinstance(personagem, Humano):
        aliado_proximo = personagem.aliadoMaisProximo(quantidade=1)  # Obter aliado mais próximo
    else:
        aliado_proximo = [0, 0]  # Lobisomem não tem aliados, então usa valores padrão

    # Para lobisomem, usa humanoMaisProximo
    if isinstance(personagem, Lobisomem):
        alvo_proximo = personagem.humanoMaisProximo()
    else:
        alvo_proximo = personagem.posLobisomem if personagem.posLobisomem and len(personagem.posLobisomem) > 0 else [0, 0]

    # Se aliado_proximo for vazio, use valores padrão [0, 0]
    if len(aliado_proximo) == 0:
        aliado_proximo = [0, 0]

    # Entradas para a rede neural
    entradas = torch.tensor([personagem.vida, personagem.fome, personagem.direcao/360, personagem.ultimo_x, personagem.ultimo_y, alvo_proximo[0], alvo_proximo[1],
                             0, 0, aliado_proximo[0], aliado_proximo[1]],
                            dtype=torch.float32)

    # Definindo a recompensa acumulada como a "saída verdadeira"
    saidas_verdadeiras = torch.tensor([[personagem.recompensa, 0.5, 45.0]], dtype=torch.float32)

    # Forward pass
    saidas_preditas = personagem.cerebro(entradas)

    # Ajustar o tamanho de saidas_preditas para [1, 3]
    saidas_preditas = saidas_preditas.unsqueeze(0)

    # Calcular o erro
    loss = criterio(saidas_preditas, saidas_verdadeiras)

    # Backpropagation e ajuste de pesos
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Resetar a recompensa após o treino
    personagem.recompensa = 0

def mutar_pesos(modelo, taxa_mutacao=0.01):
    with torch.no_grad():
        for param in modelo.parameters():
            param.add_(torch.randn(param.size()) * taxa_mutacao)


def main():
    pygame.init()
    sair = False
    clock = pygame.time.Clock()#opa
    largura, altura = 1000, 800 #800,600
    ultimo_tempo = 0
    ultimo_tempo2 = 0
    tela = Janela(largura,altura,"O Lobisomem e os humanos","black")
    tela.IniciarJanela()

    # Inicializando humanos e lobisomens
    num_lobisomens = 5
    num_personagens = 30
    num_comida_limite = 10
    lobisomens = []
    personagens = []
    comidas = []
    i = 0
    j = 0
    
    while i < num_personagens:
        humano = Humano(i, largura, altura, tela)
        personagens.append(humano)
        i += 1

    while i < num_personagens + num_lobisomens:
        lobisomem = Lobisomem(i, largura, altura, tela, forca=15)
        lobisomens.append(lobisomem)
        i += 1

    #lobisomens.append(Lobisomem(i, largura, altura, tela, forca=15))
    
    FPS = 60
    while not sair:
        clock.tick(FPS)
        tela.janela.fill((0, 0, 0))  # Cor preta
        #pygame.display.update()
        
        agora = pygame.time.get_ticks()
        
        #debug
        tela.Escrever(f"FPS:{FPS:>4}", 20, 20)
        tela.Escrever(f"Time:{agora/1000:>4.0f}", 20, 40)

        tela.Escrever(f"Fome:{personagens[0].fome:>4.0f}", 220, 20)
        tela.Escrever(f"Vida:{personagens[0].vida:>4.0f}", 220, 40)

        tela.Escrever(f"Fome:{lobisomens[0].fome:>4.0f}", 420, 20)
        tela.Escrever(f"Vida:{lobisomens[0].vida:>4.0f}", 420, 40)


        # Movimentação e decisões dos personagens
        for personagem in personagens:
            personagem.DesenharPersonagem()
            personagem.decidir()  # O humano usa a rede neural para decidir ações
            
            for personagem2 in personagens:
                if personagem == personagem2:
                    pass
                else:
                    if personagem.DentroDaVisao(personagem2):
                        personagem.adicionarPosicaoRelativa("HUMANO",personagem)
                        #personagem.ganhar_recompensa(2)

            for comida in comidas:
                comida.DesenharPersonagem()
                if comida.serComido(personagem):
                    personagem.ganhar_recompensa(10)  # Ganhar recompensa por comer
                    comidas.remove(comida)

            for lobisomem in lobisomens:
                if personagem.DentroDaVisao(lobisomem):
                    lobisomem.adicionarPosicaoRelativa("LOBISOMEM",lobisomem)
                
                lobisomem.ComerHumano(personagem)
                if lobisomem.DentroDaVisao(personagem):
                    lobisomem.adicionarPosicaoRelativa("HUMANO",personagem)
                    
                if not lobisomem.vivo and lobisomem.treinado_pos_morte == False:
                    treinar_personagem(lobisomem)  # Treina os lobisomens após a morte
                    lobisomem.salvar_cerebro()
                    #lobisomens.remove(lobisomem)

            if not personagem.vivo and personagem.treinado_pos_morte == False:
                treinar_personagem(personagem)  # Treina os personagens após a morte
                personagem.salvar_cerebro()  # Salva os pesos após o treino
                #personagens.remove(personagem)


        # Movimentação dos lobisomens
        for lobisomem in lobisomens:
            lobisomem.decidir()  # Lobisomem usa a rede neural para decidir
            lobisomem.DesenharPersonagem()

                
        if agora - ultimo_tempo > 1000:  # 1000 milissegundos = 1 segundo
            for personagem in personagens:
                personagem.checkSaude()
                #personagem.testComer()
                #if not personagem.vivo:
                    #print(f"Personagem {personagem.id} morreu na posição ({personagem.x}, {personagem.y})")
            for lobisomem in lobisomens:
                lobisomem.checkSaude()
            ultimo_tempo = agora  # Reseta o temporizador

        if agora - ultimo_tempo2 > 2000:
            if len(comidas) < num_comida_limite:
                comidas.append(Comida(j, largura, altura, tela))
                j+=1
            ultimo_tempo2 = agora

        if agora/1000 > 300:
            for personagem in personagens:
                treinar_personagem(personagem)  # Treina os personagens após a morte
                personagem.salvar_cerebro()  # Salva os pesos após o treino

            for lobisomem in lobisomens:
                treinar_personagem(lobisomem)  # Treina os lobisomens após a morte
                lobisomem.salvar_cerebro()

            break
            

        # Verificar se todos morreram e treinar
        todos_mortos = all(not p.vivo for p in personagens)
        if todos_mortos and not lobisomens[0].vivo:
            for lobisomem in lobisomens:
                treinar_personagem(lobisomem)  # Treina os lobisomens após a morte
                lobisomem.salvar_cerebro()
            break 
        
        pygame.display.update()
            
        keys = pygame.key.get_pressed()
        if keys[pygame.K_t]:
            pygame.display.toggle_fullscreen()
        if keys[pygame.K_RIGHT]:
            personagens[0].x += 1
        if keys[pygame.K_LEFT]:
            personagens[0].x -= 1
        if keys[pygame.K_UP]:
            personagens[0].y -= 1
        if keys[pygame.K_DOWN]:
            personagens[0].y += 1
            
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                #for personagem in personagens:
                #    personagem.salvar_cerebro()  # Salva os pesos ao sair
                #print("Todos foram salvos")
                melhor_personagem = max(personagens, key=lambda p: p.recompensa)

                # Copiar os pesos da rede do melhor personagem para os outros
                for personagem in personagens:
                    if personagem != melhor_personagem:
                        personagem.cerebro.load_state_dict(melhor_personagem.cerebro.state_dict())

                media_recompensa = sum(p.recompensa for p in personagens) / len(personagens)
                print(f"Recompensa média na geração {index + 1}: {media_recompensa}")
                tela.FecharJanela()
                sair = True


    tela.FecharJanela()
    pygame.quit()
    return "Finalizado."

while index < 10:
    main()
    index+=1

#print(main())
