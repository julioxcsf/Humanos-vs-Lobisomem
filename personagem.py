import random
import pygame
import math
import torch

from rede_neural_personagem import NeuralNetwork
from janela import Janela

tipo_personagem = {"COMIDA": 0, "HUMANO": 1, "LOBISOMEM" :2}

class Personagem:
    def __init__(self, num, largura, altura, obj_Janela):
        #atributos de contrucao e desenho
        self.id = num
        self.x = random.randint(5, largura - 5)
        self.y = random.randint(5, altura - 5)
        self.ultimo_x = 0
        self.ultimo_y = 0
        self.cor_RGB = [random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)]

        self.direcao = random.randint(0, 360)   # Direção inicial (em graus)
        self.raio_visao = 200                    # Raio da visão
        self.angulo_visao = 120                 # Ângulo da visão

        self.obj_Janela = obj_Janela
        self.surface = obj_Janela.janela               # Referência à janela para desenhar
        self.largura = largura
        self.altura = altura

        #atributos de personagens
        self.fome = 100 #se a fome chegar a zero, ele começa a perder vida
        self.vida = 200 
        self.velocidade = 1 #para x e y
        self.vivo = True

        cerebro = NeuralNetwork()

        #memoria do personagem
        # 1 - Posicao RELATIVA dos amigos vistos array: [ [x1,y1], ... ]
        self.posAliados = []

        # 2 - Posicao RELATIVA das Ultimas 2 comidas vistas 
        self.posComidas = []

        # 3 - ultima posicao do Lobisomem visto
        self.posLobisomem = []

        # para cada paasso ou movimento, terá uma atualizacao na posicao relativa,
        # MAS para simular uma imprecisao da memória, o valor incrementado nem sempre 
        # será o correto. exemplo: x1,y1 -> personagem.mover(+2,+3) -> x1 += 2, y1 += 2!

    def comer(self):
        if (self.fome < 100):
            self.fome += 20

    def morrer(self):
        self.vida = 0
        self.vivo = False
    
    def perderVida(self, valor):
        self.vida -= valor
        if self.vida <= 0:
            self.vida = 0
            self.vivo = False
            #print(f"ohhh não! morri em ({self.x},{self.y}). ID:{self.id}")
    
    def checkSaude(self, esfomear = 1):
        if self.fome > 0:
            self.fome -= esfomear
        if (self.fome <= 0 and self.vivo):
            self.perderVida(1/60)

    def moverPersonagem(self, dx, dy):
        if self.vivo:
            self.ultimo_x = self.x
            self.ultimo_y = self.y
            if math.fabs(dx) > self.velocidade:
                dx = self.velocidade
            if math.fabs(dy) > self.velocidade:
                dy = self.velocidade
                
            if self.x <= 5 and dx<0:
                self.x = self.largura
            elif self.x >= self.largura-5 and dx>0:
                self.x = 0
            else:
                self.x += dx
            
            if self.y <= 5 and dy<0:
                self.y = self.largura
            elif self.y >= self.altura-5 and dy>0:
                self.y = 0
            else:
                self.y += dy
                
            self.moverPosicaoRelativa(dx,dy)
            self.checkSaude(esfomear=1/120)
            
        else:
            pass

    def adicionarPosicaoRelativa(self, tipo_personagem, personagem):
        if tipo_personagem == 0:  # Comida
            self.posComidas.append([personagem.id, personagem.x - self.x, personagem.y - self.y])
        elif tipo_personagem == 1:  # Humano (aliado)
            encontrado = False
            for i, aliado in enumerate(self.posAliados):
                if aliado[0] == personagem.id:  # Se o ID já estiver na lista, atualiza a posição
                    self.posAliados[i] = [personagem.id, personagem.x - self.x, personagem.y - self.y]
                    encontrado = True
                    break
            if not encontrado:  # Se o ID não estiver na lista, adiciona um novo
                self.posAliados.append([personagem.id, personagem.x - self.x, personagem.y - self.y])
        elif tipo_personagem == 2:  # Lobisomem (inimigo)
            self.posLobisomem = [personagem.id, personagem.x - self.x, personagem.y - self.y]  # Um único lobisomem pode ser visto de cada vez

    def moverPosicaoRelativa(self,dx,dy):
        for pos in self.posAliados:
            pos[0] += dx
            pos[1] += dy

        for pos in self.posComidas:
            pos[0] += dx
            pos[1] += dy

        if self.posLobisomem:
            self.posLobisomem[1] += dx  # Atualizando a posição X
            self.posLobisomem[2] += dy  # Atualizando a posição Y
                 
    def removerPersonagem(self, tipo_personagem, personagem):
        if tipo_personagem == 1:  # Humano (aliado)
            self.posAliados = [aliado for aliado in self.posAliados if aliado[0] != personagem.id]
        elif tipo_personagem == 2:  # Lobisomem (inimigo)
            if self.posLobisomem and self.posLobisomem[0] == personagem.id:
                self.posLobisomem = []

    def DentroDaVisao(self, alvo, distancia_ataque=None, tela_debug=False, y_escrita=0):
        # Vetor direção do personagem (ajustado para o círculo trigonométrico padrão)
        dir_x = math.cos(math.radians(360 - self.direcao))
        dir_y = math.sin(math.radians(360 - self.direcao))

        # Vetor em direção ao alvo
        vetor_para_alvo_x = alvo.x - self.x
        vetor_para_alvo_y = alvo.y - self.y

        # Distância ao alvo
        distancia_alvo = math.sqrt(vetor_para_alvo_x ** 2 + vetor_para_alvo_y ** 2)
    
        if distancia_alvo <= 10:
            return 2  # Se o alvo está exatamente na posição do personagem, está na visão

        # Normaliza o vetor direção para o alvo
        vetor_para_alvo_x /= distancia_alvo
        vetor_para_alvo_y /= distancia_alvo

        # Produto escalar entre o vetor direção do personagem e o vetor para o alvo
        produto_escalar = dir_x * vetor_para_alvo_x + dir_y * vetor_para_alvo_y

        # Ângulo limite baseado no campo de visão
        cos_meio_angulo_visao = math.cos(math.radians(self.angulo_visao/2))

        # Debug opcional
        if tela_debug and self.id == 0:
            self.obj_Janela.Escrever(f"Direcao visao: {self.direcao:03}, Produto escalar: {produto_escalar:>8.4f}", 20, y_escrita)
            self.obj_Janela.Escrever(f"Cosseno do meio campo de visao: {cos_meio_angulo_visao:>8.4f}", 20, y_escrita + 20)

        # Verifica se o produto escalar está dentro do campo de visão
        if produto_escalar >= cos_meio_angulo_visao:
        # Verifica a distância
            alcance = 0
            distancia = self.raio_visao
            if distancia_alvo <= distancia:
                    alcance += 1 # está no campo de visao
            if distancia_ataque != None:
                distancia = distancia_ataque
                if distancia_alvo <= distancia:
                    alcance += 1 #está até no campo de ataque
                return alcance

        return 0 #nao está no campo de visao

    def desenhar_visao(self):
        # Desenha a área de visão como um arco ou fatia
        ret = pygame.Rect(self.x - self.raio_visao, self.y - self.raio_visao, 
                          self.raio_visao * 2, self.raio_visao * 2)
        
        # Convertendo o ângulo de direção para radianos
        #stard_angle = 0
        #end_angle = 0
        start_angle = math.radians(self.direcao - self.angulo_visao/2)
        end_angle = math.radians(self.direcao + self.angulo_visao/2)

        # Desenhando o arco (semicirculo)
        pygame.draw.arc(self.surface, (0, 255, 0), ret, start_angle, end_angle, 1)

        # Desenhando a linha do centro para os extremos do arco
        end_x = self.x + self.raio_visao * math.cos(math.radians(-self.direcao))
        end_y = self.y + self.raio_visao * math.sin(math.radians(-self.direcao))
        pygame.draw.line(self.surface, (255, 255, 0), (self.x, self.y), (end_x, end_y))
        #pygame.draw.line(self.surface, (255, 255, 0), (self.x, self.y), (end_x, end_y))

    def trocarDirecao(self):      
        # Chance de alterar 1 grau
        #if random.randint(0,10) < 6:
        #    self.direcao += 1
        #else:
        #    self.direcao -= 1
      
        # Muda a direção no sentido horário
        self.direcao -= 1  # Você pode controlar aqui o quanto quer mudar a direção por vez (exemplo: 1 grau)

        # Mantém o ângulo entre 0 e 360 graus
        if self.direcao < 0:
            self.direcao += 360

    def DesenharPersonagem(self):
        #pygame.draw.polygon(self.surface, self.cor_RGB, self.listaPontos)
        #self.trocarDirecao()
        if self.vivo:
            self.desenhar_visao()  # Chama a função para desenhar a visão
        else:
            self.cor_RGB = [0,0,125]
        # Desenha o personagem em si (pode ser um retângulo ou qualquer outra forma)
        pygame.draw.rect(self.surface, self.cor_RGB, (self.x - 5, self.y - 5, 10, 10))

    def testMove(self):
        self.moverPersonagem(random.randint(-5,5),random.randint(-5,5))

    def testComer(self):
        if random.randint(0,100) > 80:
            self.comer()

class Lobisomem(Personagem):
    def __init__(self, num, altura, largura, obj_Janela, forca=10):
        super().__init__(num, altura, largura, obj_Janela)
        self.cor_RGB = [219, 44, 0]  # Cor do lobisomem
        self.raio_visao = 300  # Raio de visão maior que o dos humanos
        self.velocidade = 3  # Velocidade maior
        self.forca = forca  # Força de ataque
        self.vida += 30  # Mais vida que um humano
        self.cerebro = NeuralNetwork()  # Rede neural do lobisomem
        self.carregar_cerebro()  # Carrega os pesos ao iniciar
        self.recompensa = 0  # Variável para acumular a recompensa
        self.treinado_pos_morte = False

    def ganhar_recompensa(self, valor):
        self.recompensa += valor

    def penalidade(self, valor):
        self.recompensa -= valor

    def salvar_cerebro(self):
        filename = f'pesos_lobisomem_{self.id}.pth'
        torch.save(self.cerebro.state_dict(), filename)
        #print(f"Pesos do lobisomem salvos em {filename}")
        self.treinado_pos_morte = True

    def carregar_cerebro(self):
        filename = f'pesos_lobisomem_{self.id}.pth'
        try:
            self.cerebro.load_state_dict(torch.load(filename, weights_only=True))
            #print(f"Pesos do lobisomem carregados de {filename}")
        except FileNotFoundError:
            print(f"Pesos para Lobisomem {self.id} não encontrados, começando do zero.")

    def morrer(self):
        """Quando o lobisomem morre, aplica uma penalidade."""
        self.vida = 0
        self.vivo = False
        self.penalidade(20)  # Penalidade quando o lobisomem morre

    def moverPersonagem(self, dx, dy):
        if self.vivo:
            if math.fabs(dx) > self.velocidade:
                dx = self.velocidade
            if math.fabs(dy) > self.velocidade:
                dy = self.velocidade
                
            if self.x <= 5 and dx<0:
                self.x = self.largura
            elif self.x >= self.largura-5 and dx>0:
                self.x = 0
            else:
                self.x += dx
            
            if self.y <= 5 and dy<0:
                self.y = self.largura
            elif self.y >= self.altura-5 and dy>0:
                self.y = 0
            else:
                self.y += dy
                
            self.moverPosicaoRelativa(dx,dy)
            self.checkSaude(esfomear=2/60)
            #self.penalidade(2/60)
            
        else:
            pass

    def decidir(self):
        """Decisão baseada na rede neural, semelhante ao humano."""
        # Coletar as entradas para a rede neural
        humano_proximo = self.humanoMaisProximo()  # [x_humano, y_humano]
        entradas = torch.tensor([self.vida, self.fome, self.direcao/360, self.ultimo_x, self.ultimo_y, humano_proximo[0], humano_proximo[1],
                                 0, 0, 0, 0], dtype=torch.float32)  # Ajuste conforme necessário
        
        # Passa as entradas pela rede neural
        saidas = self.cerebro(entradas)

        # Usar as saídas para mover o lobisomem
        dx = saidas[0].item()
        dy = saidas[1].item()
        nova_direcao = saidas[2].item()

        self.moverPersonagem(dx, dy)
        self.moverCabeca(nova_direcao/10)

    def moverCabeca(self,direcao_nova):
        self.direcao += direcao_nova
        self.checkSaude(1/60)

    def PerceberAmbiente(self):
        pass

    def humanoMaisProximo(self):
        """Retorna a posição do humano mais próximo."""
        if len(self.posAliados) == 0:
            return [0, 0]  # Se não houver humanos, retorna [0, 0]

        # Ordena os humanos pela distância
        humanos_ordenados = sorted(self.posAliados, key=lambda aliado: math.sqrt(aliado[1] ** 2 + aliado[2] ** 2))
        return [humanos_ordenados[0][1], humanos_ordenados[0][2]]  # Retorna a posição relativa do mais próximo

    def tirarFome(self, valor):
        self.fome += valor
        if self.fome >= 8:
            self.fome = 8

    def ComerHumano(self, obj_Humano):
        """Lobisomem tenta atacar e comer um humano se estiver dentro da visão."""
        if self.DentroDaVisao(obj_Humano , distancia_ataque=15) > 0:
            self.ganhar_recompensa(1/120)
            if self.DentroDaVisao(obj_Humano , distancia_ataque=15) > 1:
                if obj_Humano.vivo:
                    if obj_Humano.vida <= self.forca:
                        self.tirarFome(obj_Humano.vida)
                        obj_Humano.morrer()
                        obj_Humano.penalidade(100)
                        self.ganhar_recompensa(50)  # Ganha recompensa ao comer um humano
                    else:
                        self.tirarFome(self.forca)
                        obj_Humano.perderVida(self.forca)
                        self.ganhar_recompensa(30)  # Ganha recompensa parcial ao atacar
        else:
            self.penalidade(1/120)

    def checkSaude(self, esfomear = 4/60):#vai morrendo mais rápido, pois pode atacar
        if self.fome > 0:
            self.fome -= esfomear
            self.penalidade(1/60)
        if (self.fome <= 0 and self.vivo):
            self.perderVida(2/60) # Lobisomem perde mais vida quando está com fome

    def DesenharPersonagem(self):
        """Desenha o lobisomem na tela como um círculo."""
        if self.vivo:
            self.cor_RGB = [125,0,0]
            self.desenhar_visao()

        else:
            self.cor_RGB = [110,30,0]
        pygame.draw.circle(self.surface, [255,0,0], (self.x, self.y), 10)
        pygame.draw.circle(self.surface, self.cor_RGB, (self.x, self.y), 8)
                
    

class Humano(Personagem):
    def __init__(self, num, altura, largura, obj_Janela):
        super().__init__(num, altura, largura, obj_Janela)
        #self.alcance_visao -= 3
        self.cerebro = NeuralNetwork()
        self.recompensa = 0  # Variável para acumular a recompensa durante o jogo
        self.carregar_cerebro()  # Carrega os pesos ao iniciar
        self.treinado_pos_morte = False
        self.ultima_direcao = 0

    def ganhar_recompensa(self, valor):
        self.recompensa += valor  # Adiciona recompensa

    def penalidade(self, valor):
        self.recompensa -= valor  # Subtrai recompensa como penalidade

    def salvar_cerebro(self):
        filename = f'pesos_humano_{self.id}.pth'
        torch.save(self.cerebro.state_dict(), filename)
        #print(f"Pesos salvos em {filename}")
        self.treinado_pos_morte = True

    def carregar_cerebro(self):
        filename = f'pesos_humano_{self.id}.pth'
        try:
            self.cerebro.load_state_dict(torch.load(filename, weights_only=True))
            #print(f"Pesos carregados de {filename}")
        except FileNotFoundError:
            pass
            #print(f"Pesos para Humano {self.id} não encontrados, começando do zero.")

    def morrer(self):#sobrescreve morrer
        self.vida = 0
        self.vivo = False
        self.penalidade(20)  # Penalidade quando o personagem morre

    def perderVida(self, valor):
        self.vida -= valor
        if self.vida <= 0:
            self.vida = 0
            self.vivo = False
            #print(f"ohhh não! morri em ({self.x},{self.y}). ID:{self.id}")
            self.penalidade(10)
            
    def decidir(self):
        # Coleta as informações sobre o ambiente
        comida_proxima = self.comidaMaisProxima()  # [x_comida, y_comida]
        aliado_proximo = self.aliadoMaisProximo(quantidade=1)  # Obter 1 aliado mais próximo
        lobisomem_pos = self.posLobisomem if self.posLobisomem and len(self.posLobisomem) > 0 else [0, 0]

        # Se aliado_proximo for vazio, use valores padrão [0, 0]
        if len(aliado_proximo) == 0:
            aliado_proximo = [0, 0]

        # Evitar lobisomem
        distancia_lobisomem = math.sqrt(lobisomem_pos[0]**2 + lobisomem_pos[1]**2)
        if distancia_lobisomem < self.raio_visao:  # Se o lobisomem estiver muito perto, penaliza o personagem
            self.penalidade(2/60)
        else:
            pass
            #self.ganhar_recompensa(1/60)  # Recompensa por ficar longe

        # Prepara as entradas para a rede neural, garantindo que as listas tenham valores
        entradas = torch.tensor([self.vida, self.fome, self.direcao/360, self.ultimo_x, self.ultimo_y, comida_proxima[0], comida_proxima[1],
                             lobisomem_pos[0], lobisomem_pos[1], aliado_proximo[0], aliado_proximo[1]],
                            dtype=torch.float32)
    
        # Passa as entradas pela rede neural
        saidas = self.cerebro(entradas)

        # Usar as saídas para mover o personagem
        dx = saidas[0].item()
        dy = saidas[1].item()
        nova_direcao = saidas[2].item()

        self.moverPersonagem(dx, dy)
        self.ultima_direcao = self.direcao
        self.moverCabeca(nova_direcao/10)

    def moverCabeca(self,direcao_nova):
        self.direcao += direcao_nova
        self.checkSaude(1/60)
        
    def comidaMaisProxima(self):
        if len(self.posComidas) == 0:
            return [0, 0]  # Se não houver comida, retorna [0, 0]
    
        comida_mais_proxima = min(self.posComidas, key=lambda c: math.sqrt(c[1] ** 2 + c[2] ** 2))
        return [comida_mais_proxima[1], comida_mais_proxima[2]]  # Retorna a posição relativa

    def aliadoMaisProximo(self, quantidade=1):
        if len(self.posAliados) == 0:
            return []  # Se não houver aliados, retorna uma lista vazia
    
        # Ordena os aliados pela distância (do mais próximo ao mais distante)
        aliados_ordenados = sorted(self.posAliados, key=lambda aliado: math.sqrt(aliado[1] ** 2 + aliado[2] ** 2))
    
        # Retorna a quantidade solicitada dos aliados mais próximos
        return aliados_ordenados[:quantidade]

    def perseguirComida(self,comida_x,comida_y):
        #comida.x,comida.y = self.comudaMaisProxima()#lembrando de onde viu a comida mais proxima
        #if comida.x == 0 and comida.y == 0:
            #pass
        if False:
            pass
        else:
            if comida_x-self.x > 5:
                mover_x = 5
            elif comida_x-self.x < -5:
                mover_x = -5
            elif 0 <= comida_x-self.x <= -5:
                mover_x = comida_x-self.x
            else:
                mover_x = -1*(comida_x-self.x)

            if comida_y-self.y > 5:
                mover_y = 5
            elif comida_y-self.y < -5:
                mover_y = -5
            elif 0 <= comida_y-self.y <= -5:
                mover_y = comida_y-self.y
            else:
                mover_y = -1*(comida_y-self.y)

            self.moverPersonagem(mover_x,mover_y)

class Comida(Personagem):
    def __init__(self, num, altura, largura, obj_Janela):
        super().__init__(num, altura, largura, obj_Janela)
        self.velocidade = 0
        self.cor_RGB = [0,255,0]
        self.raio = 10

    def DesenharPersonagem(self):
        if self.vivo:
            # Desenha o personagem em si (pode ser um retângulo ou qualquer outra forma)
            #pygame.draw.rect(self.surface, self.cor_RGB, (self.x - 5, self.y - 5, 10, 10))
            pygame.draw.circle(self.surface, self.cor_RGB, (self.x, self.y), self.raio)
            pygame.draw.circle(self.surface, [255,255,0], (self.x, self.y), self.raio-1)
            pygame.draw.circle(self.surface, [255,0,0], (self.x, self.y), self.raio-3)

    def serComido(self,personagem):
        # Vetor em direção ao alvo
        distancia_x = personagem.x - self.x
        distancia_y = personagem.y - self.y

        # Distância ao alvo
        distancia_alvo = math.sqrt(distancia_x ** 2 + distancia_y ** 2)

        if distancia_alvo <= self.raio:
            personagem.comer()
            self.morrer()
            personagem.ganhar_recompensa(40)
            return True
        else:
            return False
        

        
