# 🗺️ Otimizador de Rotas Turísticas - Pesquisa Operacional

Projeto desenvolvido para a disciplina de Pesquisa Operacional (Prof. Tiago Batista Pedra), focado na implementação e análise de algoritmos de otimização combinatória aplicados a um problema real de turismo.

O sistema é implementado em Python e apresenta uma interface web interativa (Streamlit) para analisar dados e executar os algoritmos de otimização.

## 1. Fonte dos Dados

O projeto utiliza um conjunto de dados público de 20 pontos turísticos de Curitiba, obtido na plataforma Kaggle.

* **Link do Dataset:** [Kaggle.com](https://www.kaggle.com/datasets/gustavorossi14/turismocwb)
* **Arquivo:** `TurismoCWB(1).csv`

## 2. Funcionalidades e Modelos

O sistema implementa três abordagens distintas para resolver problemas de roteamento:

### 📊 Dashboard de Análise Exploratória (EDA)
Uma interface interativa para explorar os dados dos pontos turísticos, exibindo estatísticas, distribuições de custo e tempo, e correlações entre popularidade e avaliação (Spec 1.4).

### 💰 Rota por Orçamento (Heurística Gulosa)
Um algoritmo que resolve um problema de "Coleta de Prêmios" (*Prize-Collecting Problem*).
* **Objetivo:** Maximizar a **Popularidade** total da rota.
* **Restrições:** Um orçamento máximo de **Tempo (horas)** e **Custo (R$)** definido pelo usuário.
* **Método:** Uma heurística gulosa que, a cada passo, seleciona o próximo ponto que oferece o maior "score" (popularidade / custo de tempo) sem violar as restrições.

### 🚚 Otimização TSP (B&B vs. B&C)
Uma aba comparativa que resolve o Problema do Caixeiro Viajante (TSP) em subconjuntos de 10 locais usando dois métodos:

1.  **Branch and Bound (B&B) Puro:** Uma implementação manual em Python do algoritmo B&B, demonstrando os conceitos de ramificação, cálculo de limite (bound) e poda (pruning) (Spec 3.1).
2.  **Branch and Cut (B&C) via PuLP:** Uma formulação de Programação Linear Inteira (PLI) que utiliza o solver **CBC** (via PuLP). O CBC aplica um algoritmo de Branch and Cut (B&B + Cutting Planes) para encontrar a solução ótima (Spec 2.1).

## 3. Estrutura do Projeto
```
├── Turismo
    ├── algoritmos.py
    ├── app.py
    ├── requirements.txt
    ├── solver_pulp.py
    ├── TurismoCWB(1).csv
    └── README.md
```
## 4. Como Executar

### Pré-requisitos
* Python 3.9+

### 1. Instalação de Dependências
Crie um ambiente virtual e instale todas as bibliotecas necessárias. Você pode criar um arquivo `requirements.txt` com o conteúdo abaixo e executar `pip install -r requirements.txt`.

#### Conteúdo do `requirements.txt` (Spec 3.3)

Bibliotecas principais de dados e cálculo

pandas numpy

Biblioteca para calcular distância (latitude/longitude)

haversine

Bibliotecas para o Front-end e Dashboards (Spec 4.1)

streamlit altair

Bibliotecas para Análise Exploratória (EDA) e gráficos (Spec 1.4)

matplotlib seaborn

Biblioteca para Programação Linear (Branch & Cut)

pulp

Biblioteca para Testes Unitários

pytest


### 2. Execução da Aplicação
Para iniciar a interface web (Streamlit), execute:

```bash
streamlit run app.py
```
O aplicativo será aberto automaticamente no seu navegador.

5. Evidência de Validação (Testes Unitários - Spec 5.3)

# --- Spec 5.3: Testes Unitários ---
```bash
def test_calculo_tempo_viagem():
    """ Testa a função crítica de cálculo de tempo (Spec 5.3) """
    
    # Cenário 1: 25 km a 25 km/h deve levar 60 minutos
    tempo = alg.calculate_travel_time(dist_km=25, avg_speed_kmh=25)
    assert tempo == 60.0
    
    # Cenário 2: 10 km a 40 km/h deve levar 15 minutos (0.25h)
    tempo = alg.calculate_travel_time(dist_km=10, avg_speed_kmh=40)
    assert tempo == 15.0

    # Cenário 3: 0 km deve levar 0 minutos
    tempo = alg.calculate_travel_time(dist_km=0, avg_speed_kmh=50)
    assert tempo == 0.0```

@pytest.fixture
def dados_carregados():
    """ Fixture para carregar os dados uma vez para os testes """
    df, all_nodes, id_map, dist_matrix = alg.load_data()
    return df, all_nodes, id_map, dist_matrix

def test_carregamento_dados(dados_carregados):
    """ Testa a função de carregamento (Spec 5.3) """
    df, all_nodes, id_map, dist_matrix = dados_carregados
    
    # Verifica se carregou todos os 20 pontos
    assert len(df) == 20
    assert len(all_nodes) == 20
    assert len(dist_matrix) == 20
    
    # Verifica se o Jardim Botânico (ID 1) foi mapeado para o índice 0
    assert id_map[1] == 0
    
    # Verifica se a diagonal da matriz de distância é 0 (distância para si mesmo)
    assert dist_matrix[0][0] == 0
    assert dist_matrix[5][5] == 0

def test_heuristica_orcamento_limite_zero(dados_carregados):
    """ Testa a heurística de orçamento (função crítica) com limites baixos (Spec 5.3) """
    df, all_nodes, id_map, dist_matrix = dados_carregados

    # Teste com orçamento de tempo ZERO (só deve falhar)
    route, summary, log = alg.solve_budget_route_heuristic(
        all_nodes, dist_matrix, id_map, 
        max_time_min=0, max_cost=100, start_node_id=1
    )
    # O Jardim Botânico (ID 1) custa 90 min.
    assert len(route) == 0 # Não deve conseguir adicionar nem o ponto de partida

    # Teste com orçamento de custo ZERO
    route, summary, log = alg.solve_budget_route_heuristic(
        all_nodes, dist_matrix, id_map, 
        max_time_min=500, max_cost=0, start_node_id=1
    )
    # O Jardim Botânico (ID 1) custa R$ 15.
    assert len(route) == 0 # Não deve conseguir adicionar nem o ponto de partida
```
## 2.1 Modelagem Formal dos Problemas (Spec 2.1)

O projeto aborda dois problemas de otimização distintos, cada um com sua própria modelagem matemática.

### A. Rota TSP (Branch and Bound / Branch and Cut)

Este é um Problema do Caixeiro Viajante (TSP - *Traveling Salesperson Problem*).
```
* **Variáveis de Decisão:**
    * $x_{ij} = 1$ se a rota vai diretamente do ponto $i$ ao ponto $j$.
    * $x_{ij} = 0$ caso contrário.
    * $u_i$ = Variável auxiliar inteira para eliminação de sub-rota (usada na formulação PuLP/MTZ).

* **Função Objetivo (Minimização):**
    * Minimizar a distância total percorrida. Onde $d_{ij}$ é a distância (custo) entre $i$ e $j$.
    * $$\min Z = \sum_{i=0}^{N-1} \sum_{j=0}^{N-1} d_{ij} \cdot x_{ij}$$

* **Restrições:**
    1.  **Sair de cada ponto 1x:** $\sum_{j \neq i} x_{ij} = 1, \forall i$ (Cada ponto deve ter exatamente uma saída).
    2.  **Chegar em cada ponto 1x:** $\sum_{i \neq j} x_{ij} = 1, \forall j$ (Cada ponto deve ter exatamente uma chegada).
    3.  **Eliminação de Sub-rotas (MTZ):** $u_i - u_j + N \cdot x_{ij} \le N - 1, \forall i,j > 0, i \neq j$ (Garante um *tour* único e conectado).
    4.  **Integridade:** $x_{ij} \in \{0, 1\}$ (As decisões são binárias).

### B. Rota por Orçamento (Heurística Gulosa)

Este é um Problema de Coleta de Prêmios (*Prize-Collecting Problem*).

* **Variáveis de Decisão:**
    * $y_i = 1$ se o ponto $i$ é visitado.
    * $y_i = 0$ caso contrário.

* **Função Objetivo (Maximização):**
    * Maximizar o *score* de popularidade total dos pontos visitados. Onde $p_i$ é a popularidade do ponto $i$.
    * $$\max Z = \sum_{i=1}^{N} p_i \cdot y_i$$
    *(Nota: A heurística implementada usa um *score* ponderado (popularidade / tempo) para tomar decisões gulosas.)*

* **Restrições:**
    1.  **Orçamento de Custo:** $\sum_{i=1}^{N} c_i \cdot y_i \le C_{max}$ (O custo total de entrada $c_i$ não pode exceder o custo máximo $C_{max}$).
    2.  **Orçamento de Tempo:** $\sum_{i=1}^{N} (t_{visita,i} + t_{desloc,i}) \cdot y_i \le T_{max}$ (O tempo total (visita + deslocamento) não pode exceder o tempo máximo $T_{max}$).
    3.  **Integridade:** $y_i \in \{0, 1\}$ (A visita é binária).
```
