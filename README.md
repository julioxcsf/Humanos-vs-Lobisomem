# Humanos-vs-Lobisomem (Python + Pygame + PyTorch)

Simulação 2D de sobrevivência entre humanos e lobisomens em um ambiente hostil!  
O jogo combina elementos de **movimentação**, **percepção ambiental** e **aprendizado de máquina** para treinar os personagens usando **redes neurais**.

Desenvolvido totalmente em **Python**, utilizando:
- **Pygame** para renderização gráfica e lógica de ambiente
- **PyTorch** para redes neurais e treinamento dos personagens

---

##  Descrição

- **Humanos** tentam sobreviver encontrando comida e evitando lobisomens.
- **Lobisomens** caçam humanos para se alimentar e sobreviver.
- Ambos usam **redes neurais simples** para tomar decisões baseadas em:
  - Fome
  - Vida
  - Direção
  - Últimas posições vistas de inimigos, aliados e alimentos.

O jogo implementa **recompensas** e **punições** durante a simulação para **treinar** as redes em tempo real.

---

## 🛠 Tecnologias utilizadas

- [Python 3](https://www.python.org/)
- [Pygame](https://www.pygame.org/news)
- [PyTorch](https://pytorch.org/)
- Programação Orientada a Objetos (POO)

---

## Funcionalidades principais

- Treinamento online de redes neurais para cada personagem
- Sistema de recompensa/punição baseado em ações (comer, sobreviver, atacar)
- Campo de visão dinâmico (raio e ângulo de percepção)
- Renderização e movimentação fluida no Pygame
- Salvamento e carregamento dos pesos das redes neurais
- Evolução das decisões de movimentação e sobrevivência ao longo das gerações

---

## 📷 Demonstração
![image](https://github.com/user-attachments/assets/57473e7f-5ba9-4a13-b95d-181f28b34da8)

> *Campo de visão do lobo, comidas dos humanos e os mortos (quadrados azuis)*
