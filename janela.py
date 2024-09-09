import pygame
import math

#precisa do pygame.init() na main

# Definindo a fonte e tamanho




class Janela():
    def __init__(self,largura,altura,titulo,cor):
        self.largura = largura
        self.altura = altura
        self.titulo = titulo
        self.janela = None
        self.cor = cor
        self.font = self.DefinirFont()
        
    def IniciarJanela(self):
        self.janela = pygame.display.set_mode((self.largura,self.altura))
        pygame.display.set_caption(self.titulo)
        self.janela.fill(self.cor)

    def FecharJanela(self):
        pygame.quit()

    def TelaCheia(self):
        pygame.display.toggle_fullscreen()

    def DefinirFont(self,tamanho_font = 16): #sem alteracao de font implementada
        return pygame.font.Font(None, tamanho_font)
    
    def Escrever(self,text, x, y, color=(255, 255, 255)):
    # Renderiza o texto
        text_surface = self.font.render(text, True, color)
    
    # Desenha o texto na tela na posição (x, y)
        self.janela.blit(text_surface, (x, y))


class Entidade():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.velx = 0
        self.vely = 0
        self.angulo = 0

class Poligono(Entidade):
    def __init__(self,cor,obj_Janela):#tinha listaPontos
        self.listaPontos = []
        self.cor = cor #[R,G,B]
        self.estadoCor = 1
        self.surface = obj_Janela

    def Clarear(self):
        if self.cor[0] <= 235:
            self.cor[0] += 20
            
        if self.cor[0] > 235:
            self.cor[0] = 255

        if self.cor[1] <= 235:
            self.cor[1] += 20

        if self.cor[1] > 235:
            self.cor[1] = 255

        if self.cor[2] <= 235:
            self.cor[2] += 20

        if self.cor[2] > 235:
            self.cor[2] = 255

    def Escurecer(self):
        if self.cor[0] >= 20:
            self.cor[0] -= 20
            
        if self.cor[0] < 20:
            self.cor[0] = 0

        if self.cor[1] >= 20:
            self.cor[1] -= 20

        if self.cor[1] < 20:
            self.cor[1] = 0

        if self.cor[2] >= 20:
            self.cor[2] -= 20

        if self.cor[2] < 20:
            self.cor[2] = 0

    def Colorindo(self):
        r=self.cor[0]
        g=self.cor[1]
        b=self.cor[2]
        if r+g!=255 and g+b!=255 and r+b!=255:
            r,g,b = 255,0,0
            
        if self.estadoCor==1:
            r -= 1
            g += 1
            if r <= 0:
                self.estadoCor = 2
                
        elif self.estadoCor == 2:
            g -= 1
            b += 1
            if g<= 0:
                self.estadoCor = 3
        else:
            b -= 1
            r += 1
            if b <= 0 :
                self.estadoCor = 1
                
        self.cor = (r,g,b)
                

    def Rotacionar(self,pontoRef,angulo):#em graus
        self.angulo = angulo
        angulo = math.radians(angulo)
        for ponto in range(len(self.listaPontos)):
            x = self.listaPontos[ponto][0] - pontoRef[0]
            y = self.listaPontos[ponto][1] - pontoRef[1]

            x1 = x * math.cos(angulo) + y * math.sin(angulo)#-
            y1 = x * math.sin(angulo) - y * math.cos(angulo)#+

            self.listaPontos[ponto][0] = x1 + pontoRef[0]
            self.listaPontos[ponto][1] = y1 + pontoRef[1]

    def DesenharPoligono(self):
        pygame.draw.polygon(self.surface, self.cor, self.listaPontos)


class Botao(Poligono):
    def __init__(self,ponto,largura,altura):#basicamente um retangulo
        pygame.mouse.get_pressed(num_buttons=3)
        pygame.mouse.get_pos()#(x,y)

    def click(self):
        if self.x <= pygame.mouse.get_pos()[0] <=self.x + self.largura:
            if self.y <= pygame.mouse.get_pos()[1] <=self.y + self.altura:
                self.Clarear()
                if pygame.mouse.get_pressed(num_buttons=3)[0]:
                    self.Escurecer()
                    return True
