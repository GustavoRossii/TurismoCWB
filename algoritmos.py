# Este arquivo deve ser salvo como: algoritmos.py

import pandas as pd
import numpy as np
from haversine import haversine, Unit
import time
import sys # Mantido por precaução, embora o setrecursionlimit tenha sido removido

# --- Configurações Globais ---
# CORREÇÃO 1: Mudei 25 para 25.0 para evitar erro de tipo no Streamlit
AVG_SPEED_KMH = 25.0  # Velocidade média estimada para deslocamento em Curitiba
CSV_FILE = 'TurismoCWB(1).csv'

# --- Variáveis Globais para o B&B (Spec 3.2) ---
# (Nenhuma alteração nesta classe)
class BnBStats:
    def __init__(self):
        self.upper_bound = float('inf')
        self.best_path = []
        self.nodes_expanded = 0
        self.pruning_count = 0
        self.start_time = 0
        self.end_time = 0

    def reset(self):
        self.upper_bound = float('inf')
        self.best_path = []
        self.nodes_expanded = 0
        self.pruning_count = 0
        self.start_time = 0
        self.end_time = 0

    def get_results(self):
        return {
            "cost": self.upper_bound,
            "path": self.best_path,
            "nodes": self.nodes_expanded,
            "pruned": self.pruning_count,
            "time": self.end_time - self.start_time
        }

# =============================================================================
# FUNÇÕES DE DADOS E CÁLCULO
# =============================================================================

def load_data():
    """
    Carrega, limpa e prepara os dados do CSV.
    Retorna o DataFrame completo e o mapeamento ID -> Índice.
    """
    try:
        df = pd.read_csv(CSV_FILE)
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Spec 1.2: Limpeza e padronização
        df['custo_entrada'] = df['custo_entrada'].fillna(0)
        df['tempo_visita_min'] = df['tempo_visita_min'].fillna(df['tempo_visita_min'].median())
        
        all_nodes_data = df.to_dict('records')
        id_to_index = {node['id']: i for i, node in enumerate(all_nodes_data)}
        
        # Calcular matriz de distância completa
        dist_matrix_full = calculate_distance_matrix(all_nodes_data)

        return df, all_nodes_data, id_to_index, dist_matrix_full

    except FileNotFoundError:
        print(f"Erro: Arquivo '{CSV_FILE}' não encontrado.")
        return None, None, None, None
    except Exception as e:
        print(f"Erro ao ler o CSV: {e}")
        return None, None, None, None

def calculate_distance_matrix(nodes):
    """
    Calcula a matriz de distâncias (custos) entre todos os pontos 
    usando a fórmula Haversine com lat/lon.
    """
    n = len(nodes)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                loc_i = (nodes[i]['latitude'], nodes[i]['longitude'])
                loc_j = (nodes[j]['latitude'], nodes[j]['longitude'])
                dist_matrix[i][j] = haversine(loc_i, loc_j, unit=Unit.KILOMETERS)
    return dist_matrix

def calculate_travel_time(dist_km, avg_speed_kmh=AVG_SPEED_KMH):
    """Calcula o tempo de viagem em minutos."""
    if avg_speed_kmh <= 0:
        return float('inf')
    return (dist_km / avg_speed_kmh) * 60  # Converte horas para minutos

# =============================================================================
# PARTE 1: ALGORITMO BRANCH AND BOUND PARA TSP (Spec 3.1)
# =============================================================================

def _calculate_heuristic_upper_bound(dist_matrix):
    """
    Spec 5.1: Heurística Simples (Vizinho Mais Próximo)
    """
    n = len(dist_matrix)
    current_node = 0
    path = [current_node]
    visited = {current_node}
    total_cost = 0

    while len(visited) < n:
        last_node = path[-1]
        nearest_neighbor = -1
        min_dist = float('inf')

        for neighbor in range(n):
            if neighbor not in visited:
                if dist_matrix[last_node][neighbor] < min_dist:
                    min_dist = dist_matrix[last_node][neighbor]
                    nearest_neighbor = neighbor
        
        if nearest_neighbor != -1:
            total_cost += min_dist
            path.append(nearest_neighbor)
            visited.add(nearest_neighbor)
        else:
            break
            
    total_cost += dist_matrix[path[-1]][current_node]
    path.append(current_node)
    
    return total_cost, path

def _solve_tsp_branch_and_bound(dist_matrix, stats):
    """
    Spec 3.1: Implementação do Algoritmo Branch and Bound
    Recebe um objeto 'stats' para atualizar.
    """
    n = len(dist_matrix)
    stack = [ ([0], 0) ] 

    while stack:
        current_path, current_cost = stack.pop()
        
        stats.nodes_expanded += 1
        
        if len(current_path) == n:
            cost_to_complete = dist_matrix[current_path[-1]][0]
            final_cost = current_cost + cost_to_complete
            
            if final_cost < stats.upper_bound:
                stats.upper_bound = final_cost
                stats.best_path = current_path + [0]
            continue

        last_node = current_path[-1]
        for next_node in range(n):
            if next_node not in current_path:
                
                new_path = current_path + [next_node]
                new_cost = current_cost + dist_matrix[last_node][next_node]
                lower_bound = new_cost 

                if lower_bound < stats.upper_bound:
                    stack.append( (new_path, new_cost) )
                else:
                    stats.pruning_count += 1

def run_tsp_experiment(experiment_name, nodes_data):
    """
    Função wrapper para rodar um experimento TSP B&B.
    Retorna o nome, as métricas e o caminho.
    """
    # CORREÇÃO 2: Removida a restrição de "!= 10"
    # Agora aceita qualquer número de nós (desde que >= 2)
    if len(nodes_data) < 2:
        print("Erro B&B: Pelo menos 2 nós são necessários.")
        return None
    
    # CORREÇÃO 3: Removido o sys.setrecursionlimit, pois
    # a implementação _solve_tsp_branch_and_bound é iterativa (usa stack)
    # e não recursiva.
    
    index_to_name = {i: node['nome'] for i, node in enumerate(nodes_data)}
    dist_matrix = calculate_distance_matrix(nodes_data)
    
    stats = BnBStats()
    
    # Calcular Limite Superior Inicial (Heurística)
    heuristic_cost, heuristic_path = _calculate_heuristic_upper_bound(dist_matrix)
    stats.upper_bound = heuristic_cost
    stats.best_path = heuristic_path
    
    # Rodar Branch and Bound
    stats.start_time = time.time()
    _solve_tsp_branch_and_bound(dist_matrix, stats)
    stats.end_time = time.time()

    # Formatar resultados
    results = stats.get_results()
    results['name'] = experiment_name
    results['heuristic_cost'] = heuristic_cost # Inclui o custo da heurística
    results['path_names'] = " -> ".join([index_to_name[idx] for idx in results['path']])
    
    return results

# =============================================================================
# PARTE 2: ALGORITMO HEURÍSTICO PARA ROTA COM ORÇAMENTO (Spec 5.1)
# =============================================================================

def solve_budget_route_heuristic(all_nodes, dist_matrix_full, id_to_index, max_time_min, max_cost, start_node_id=1):
    """
    Implementa uma heurística gulosa (Spec 5.1) para o problema de 
    orçamento (Prize Collecting).
    """
    
    route = []
    route_cost = 0.0
    route_time = 0.0
    route_popularity = 0.0
    log_messages = [] # Log para exibir no Streamlit
    
    try:
        current_node_idx = id_to_index[start_node_id]
        start_node_data = all_nodes[current_node_idx]
    except KeyError:
        log_messages.append(f"Erro: Nó inicial (ID {start_node_id}) não encontrado.")
        return [], {}, log_messages

    # 1. Adicionar o ponto de partida
    if start_node_data['tempo_visita_min'] <= max_time_min and start_node_data['custo_entrada'] <= max_cost:
        route.append(start_node_data)
        route_cost = start_node_data['custo_entrada']
        route_time = start_node_data['tempo_visita_min']
        route_popularity = start_node_data['popularidade']
        visited_ids = {start_node_id}
        log_messages.append(f"Ponto de partida: {start_node_data['nome']} (Custo: R${route_cost}, Tempo: {route_time} min)")
    else:
        log_messages.append(f"Ponto de partida ({start_node_data['nome']}) excede o orçamento. Rota vazia.")
        return [], {}, log_messages

    # 2. Loop Guloso
    while True:
        best_candidate = None
        best_score = -1
        
        last_node_idx = id_to_index[route[-1]['id']]

        for candidate_node in all_nodes:
            if candidate_node['id'] not in visited_ids:
                
                candidate_idx = id_to_index[candidate_node['id']]
                
                travel_dist = dist_matrix_full[last_node_idx][candidate_idx]
                travel_time = calculate_travel_time(travel_dist, AVG_SPEED_KMH)
                visit_time = candidate_node['tempo_visita_min']
                visit_cost = candidate_node['custo_entrada']

                time_if_added = route_time + travel_time + visit_time
                cost_if_added = route_cost + visit_cost
                
                if time_if_added <= max_time_min and cost_if_added <= max_cost:
                    # Função Objetivo (Heurística)
                    time_cost = travel_time + visit_time + 1 
                    score = candidate_node['popularidade'] / time_cost
                    
                    if score > best_score:
                        best_score = score
                        best_candidate = candidate_node
        
        # 3. Adicionar o melhor candidato
        if best_candidate:
            candidate_idx = id_to_index[best_candidate['id']]
            travel_dist = dist_matrix_full[last_node_idx][candidate_idx]
            travel_time = calculate_travel_time(travel_dist, AVG_SPEED_KMH)
            
            route_time += travel_time + best_candidate['tempo_visita_min']
            route_cost += best_candidate['custo_entrada']
            route_popularity += best_candidate['popularidade']
            visited_ids.add(best_candidate['id'])
            route.append(best_candidate)
            
            log_messages.append(f"  -> Adicionando: {best_candidate['nome']} (Dist: {travel_dist:.1f}km, Tempo Viagem: {travel_time:.0f}min)")
            
        else:
            log_messages.append("\nNenhum outro ponto pôde ser adicionado respeitando o orçamento.")
            break
            
    # 4. Preparar sumário
    summary = {
        "score_popularidade": route_popularity,
        "tempo_total_gasto": route_time,
        "custo_total_gasto": route_cost,
        "tempo_max": max_time_min,
        "custo_max": max_cost,
        "path_names": " -> ".join([node['nome'] for node in route])
    }
    
    return route, summary, log_messages