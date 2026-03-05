**O Problema da Semana: "A Tríade do Big Data"**
 
**Contexto:**
 
 Nossa empresa de IA cresceu. Agora, além de instâncias para Deep Learning (DL) e Data Analytics (DA), fomos contratados para rodar rotinas pesadas de Engenharia de Dados / ETL (ED).
 
 **Dados do Problema (Lucro e Consumo por hora):**
 
 **DL ($x_1$)**: Lucro R$ 50,00 | Consome: 2 CPU, 1 GB RAM, 0 TB Storage.
 
 **DA ($x_2$)**: Lucro R$ 30,00 | Consome: 1 CPU, 2 GB RAM, 1 TB Storage.
 
 **ED ($x_3$)**: Lucro R$ 40,00 | Consome: 1 CPU, 1 GB RAM, 2 TB Storage.
 
 **Recursos Físicos do Cluster:**
 
 **CPU**: Máximo de 150 unidades.
 
 **RAM**: Máximo de 160 GB.
 
 **Storage** (Armazenamento): Máximo de 100 TB.

 Para simplificar o Simplex inicial, retiramos as restrições contratuais de mínimo.

 1. **Formulação**: Escreva a Função Objetivo e as Inequações de restrição no caderno.
 
 2. **A Forma Padrão (Hillier Cap 4.2)**: 
 Introduza as Variáveis de Folga ($s_1, s_2, s_3$) para transformar todas as restrições $\leq$ em igualdades ($=$)
 3. **Reflexão:** Com 3 variáveis e 3 restrições, é impossível desenhar o gráfico no papel de forma simples. Você precisará do Simplex.

**O que faremos na Aula Prática:**
No laboratório, vamos resolver esse problema em Python e focar em ler a mente do Simplex. Vamos aprender a extrair do Python não apenas a resposta final, mas o valor de cada Variável de Folga ($s$), para responder ao gerente de TI: "Qual servidor precisará de upgrade primeiro: CPU, RAM ou Storage?".


