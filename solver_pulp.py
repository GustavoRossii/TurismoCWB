# Este arquivo deve ser salvo como: solver_pulp.py

import pulp
import time
import algoritmos as alg # Reutiliza nosso carregador de dados e matriz de distância

def solve_tsp_with_pulp(experiment_name, nodes_data):
    """
    Resolve o TSP usando Programação Linear Inteira (PuLP).
    Isto utiliza um solver que aplica Branch and Cut (B&B + Cutting Plane).
    
    Usamos a formulação Miller-Tucker-Zemlin (MTZ) para eliminar sub-rotas.
    """
    
    print(f"\n--- Iniciando Solver PuLP (Branch & Cut) para: {experiment_name} ---")
    
    if len(nodes_data) < 2:
        return None

    # 1. Preparar dados
    n = len(nodes_data)
    index_to_name = {i: node['nome'] for i, node in enumerate(nodes_data)}
    
    # Reutiliza nossa função para calcular a matriz de distâncias
    dist_matrix = alg.calculate_distance_matrix(nodes_data)

    # 2. Modelagem do Problema (Spec 2.1)
    
    # --- a. Definir o Problema ---
    # Queremos minimizar a distância
    prob = pulp.LpProblem(f"TSP_{experiment_name}", pulp.LpMinimize)

    # --- b. Variáveis de Decisão ---
    
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), cat=pulp.LpBinary)
    u = pulp.LpVariable.dicts("u", range(n), lowBound=1, upBound=n, cat=pulp.LpInteger)
    cost = pulp.lpSum(
        dist_matrix[i][j] * x[i][j] 
        for i in range(n) 
        for j in range(n) 
        if i != j
    )
    prob += cost
    for i in range(n):
        prob += pulp.lpSum(x[i][j] for j in range(n) if i != j) == 1

    # R2: Entrar em cada cidade exatamente uma vez
    for j in range(n):
        prob += pulp.lpSum(x[i][j] for i in range(n) if i != j) == 1

    # R3: Restrição de não ir de i para i (diagonal)
    for i in range(n):
        prob += x[i][i] == 0

    # R4: Restrições de Eliminação de Sub-rota (MTZ)
    # Estes são os "Planos de Corte" (Cutting Planes) da formulação
    # u_i - u_j + n * x_ij <= n - 1   (para i, j > 0 e i != j)
    for i in range(1, n): # Começa do nó 1 (0 é o depósito)
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i][j] <= n - 1

    # 3. Executar o Solver
    print("Iniciando o solver... (Isso pode levar alguns segundos/minutos)")
    start_time = time.time()
    
    # status = prob.solve() # Usa o solver padrão (CBC)
    # Para problemas de TSP, o solver GLPK pode ser mais rápido se instalado
    # Tenta usar GLPK, se não, usa o padrão CBC
    try:
        status = prob.solve(pulp.GLPK_CMD(msg=0))
    except pulp.apis.core.PulpSolverError:
        print("GLPK não encontrado, usando o solver padrão (CBC).")
        status = prob.solve()
        
    end_time = time.time()
    
    if status != pulp.LpStatusOptimal:
        print("!!! O Solver não encontrou uma solução ótima !!!")
        return None

    # 4. Recuperar a Solução
    path = [0]
    current_node = 0
    while len(path) < n:
        for j in range(n):
            # Encontra o próximo nó j para onde x_ij é 1
            if j != current_node and pulp.value(x[current_node][j]) == 1:
                path.append(j)
                current_node = j
                break
    path.append(0) # Volta ao início

    # 5. Formatar Resultados
    results = {
        "name": f"{experiment_name} (PuLP)",
        "cost": pulp.value(prob.objective),
        "path": path,
        "path_names": " -> ".join([index_to_name[idx] for idx in path]),
        "time": end_time - start_time,
        "solver_status": pulp.LpStatus[status]
    }
    
    print(f"Solução Ótima (PuLP) encontrada: {results['cost']:.2f} km")
    print(f"Tempo de execução (PuLP): {results['time']:.4f} s")
    
    return results

if __name__ == '__main__':
    # Bloco de teste
    print("Iniciando teste do solver PuLP...")
    
    # Carrega os dados (função do algoritmos.py)
    df, all_nodes, id_to_index, dist_matrix_full = alg.load_data()
    
    if df is not None:
        # Pega os 10 primeiros para o teste
        primeiros_10 = all_nodes[0:10]
        
        # Roda o B&B puro
        results_bnb = alg.run_tsp_experiment("10 Primeiros (B&B Puro)", primeiros_10)
        
        # Roda o PuLP (Branch & Cut)
        results_pulp = solve_tsp_with_pulp("10 Primeiros (PuLP)", primeiros_10)
        
        if results_bnb and results_pulp:
            print("\n--- Comparação dos Resultados ---")
            print(f"B&B Puro:   {results_bnb['cost']:.2f} km em {results_bnb['time']:.4f} s")
            print(f"PuLP (B&C): {results_pulp['cost']:.2f} km em {results_pulp['time']:.4f} s")
        else:
            print("Não foi possível executar os dois algoritmos para comparação.")
            
    else:
        print("Erro ao carregar dados.")