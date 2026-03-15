**O Problema da Semana: "Acompanhando as mudanças"**
 
**Contexto:**

Voltamos ao nosso cluster de IA (Problema da Tríade do Big Data da Semana 3). 

Sabemos que a CPU é o nosso limite (temos 150 unidades).

Hoje, o provedor de infraestrutura (ex: AWS ou Azure) ofereceu um "Burst Rate" (expansão de emergência): eles podem nos alugar um pacote extra de **10 CPUs adicionais** por um custo fixo de **R$ 250,00** para as próximas horas.

Antes de qualquer cálculo, o que você acha que devemos fazer? Qual sua intuição sobre essa oferta?

 **Dados do Problema (Lucro e Consumo por hora):**
 
 **DL ($x_1$)**: Lucro R$ 50,00 | Consome: 2 CPU, 1 GB RAM, 0 TB Storage.
 
 **DA ($x_2$)**: Lucro R$ 30,00 | Consome: 1 CPU, 2 GB RAM, 1 TB Storage.
 
 **ED ($x_3$)**: Lucro R$ 40,00 | Consome: 1 CPU, 1 GB RAM, 2 TB Storage.
 
 **Recursos Físicos do Cluster:**
 
 **CPU**: Máximo de 150 unidades.
 
 **RAM**: Máximo de 160 GB.
 
 **Storage** (Armazenamento): Máximo de 100 TB.

 Lembrando que nós já encontramos a solução que oferece o maior lucro para o nosso problema e que a CPU é o nosso recurso escasso: 

 - Lucro de R$ 4.500,00 produzindo 50 VMs de DL e 50 de ED.
 - A CPU esgotou completamente (Folga = 0)
 - O armazenamento (storage) esgotou completamente (Folga = 0)
 - Temos memória sobrando (Folga = 60)


**Desafio extra:**

Considerando os resultados obtidos qual a decisão a tomar?

Há algo que possa impactar positivamente o resultado?


**Soluções**

Na pasta você irá encontrar 4 soluções para o problema da semana:
- resposta_aula_04a.py - Solução utilizando o preço sombra para avaliar a oferta
- resposta_aula_04b.py - Solução realizada como se tivessemos as 10 cpus adicionais para comparar o valor do resultado obtido
- resposta_aula_04c.py - Contraproposta com 10Gb de storage adicionais, além das 10 CPUs com o custo de R$250,00 pelo pacote
- resposta_aula_04d.py - Avaliação do problema considerando o aumento do valor das instâncias de Data Analytics para R$45,00. Nesse caso, há alteração do consumo dos recursos e dos preços-sombra, gerando novo contexto para decisão.