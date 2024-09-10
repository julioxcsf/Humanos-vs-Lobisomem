# Humanos-vs-Lobisomem
Algo semelhante a um jogo onde humanos competem pela comida no terreno e fogem do Lobisomem.

Funcionamento:
Os humanos e o Lobisomem têm fome e precisam buscar comida. O lobisomem busca o humano e o humano foge enquanto busca a comida solta no terreno.
Ambos podem ver os outros personagens e lembrar de suas posições relativas a si quando o personagem visto ja não está mais em vista.
A fome é um contador que quando chega a zero, o personagem começa a perder vida. Ganha quem sobreviver até o final.
Cada personagem tem memoria do que viu(x e y) e essa memoria é atualizada conforme ele se move pelo terreno.
A rede neural de cada personagem recebe como entrada:
  1. a fome do personagem;
  2. a vida do personagem;
  3. a direcao da visao;
  4. a sua ultima posicao (x e y);
  5. a posição relativa da ultima comida vista (x e y);
  6. posicao relativa do Lobisomem (x e y);
  7. posicao relativa dos outros Humanos(x e y do ultimo visto);

