import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import algoritmos as alg
import solver_pulp as pulp_solver 

# --- Configuração da Página ---
st.set_page_config(
    page_title="Otimizador de Rotas Turísticas - CWB",
    page_icon="🗺️",
    layout="wide"
)

# --- Carregamento de Dados (Cache) ---
@st.cache_data
def load_data_cached():
    """ Carrega os dados e armazena em cache para performance."""
    df, all_nodes, id_map, dist_matrix = alg.load_data()
    if df is None:
        st.error(f"Erro fatal ao carregar o arquivo '{alg.CSV_FILE}'. Verifique se o arquivo está na pasta.")
        st.stop()
    return df, all_nodes, id_map, dist_matrix

# Carrega os dados uma vez
df, all_nodes, id_to_index, dist_matrix_full = load_data_cached()


# --- Título Principal ---
st.title("🗺️ Sistema de Otimização de Rotas Turísticas - Curitiba")
st.write("Projeto de Pesquisa Operacional - Prof. Tiago Batista Pedra")

# --- Definição das Abas ---
tab_eda, tab_budget, tab_tsp = st.tabs([
    "📊 Análise Exploratória (EDA)",
    "💰 Rota por Orçamento (Heurística)",
    "🚚 Otimização TSP (B&B vs PuLP)" # Título da aba atualizado
])


# =============================================================================
# ABA 1: ANÁLISE EXPLORATÓRIA (Spec 1.4 e 4.2)
# =============================================================================
with tab_eda:
    st.header("Análise Exploratória dos Pontos Turísticos")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Pontos", f"{len(df)}")
    col2.metric("Custo Médio Entrada", f"R$ {df['custo_entrada'].mean():.2f}")
    col3.metric("Tempo Médio Visita", f"{df['tempo_visita_min'].mean():.0f} min")
    col4.metric("Avaliação Média", f"{df['avaliacao'].mean():.1f} ⭐")
    
    st.divider()
    
    # Gráficos
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribuição de Custos de Entrada")
        chart_custo = alt.Chart(df).mark_bar().encode(
            x=alt.X('custo_entrada:Q', bin=True, title='Custo da Entrada (R$)'),
            y=alt.Y('count()', title='Contagem de Locais'),
            tooltip=['custo_entrada', 'count()']
        ).interactive()
        st.altair_chart(chart_custo, use_container_width=True)
        
        st.subheader("Popularidade vs. Avaliação")
        chart_pop_aval = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('avaliacao:Q', title='Avaliação (0-5)'),
            y=alt.Y('popularidade:Q', title='Popularidade (0-100)'),
            color='categoria',
            tooltip=['nome', 'avaliacao', 'popularidade', 'categoria']
        ).interactive()
        st.altair_chart(chart_pop_aval, use_container_width=True)

    with c2:
        st.subheader("Tempo de Visita por Categoria")
        chart_tempo_cat = alt.Chart(df).mark_boxplot().encode(
            x=alt.X('categoria:N', title='Categoria'),
            y=alt.Y('tempo_visita_min:Q', title='Tempo de Visita (min)'),
            tooltip=['categoria', 'tempo_visita_min']
        ).interactive()
        st.altair_chart(chart_tempo_cat, use_container_width=True)

    st.subheader("Tabela de Dados Completa")
    st.dataframe(df)


# =============================================================================
# ABA 2: ROTA POR ORÇAMENTO (Spec 4.1, 4.3, 4.4)
# =============================================================================
with tab_budget:
    st.header("Planejador de Rota por Orçamento")
    st.markdown("Defina seu orçamento de tempo e custo para encontrar a melhor rota (maximizando popularidade), **partindo do Jardim Botânico**.")

    # --- Inputs do Usuário (Spec 4.3) ---
    col1, col2 = st.columns(2)
    with col1:
        user_budget_horas = st.slider(
            "Quantas horas você tem disponível?", 
            min_value=1.0, max_value=24.0, value=8.0, step=0.5
        )
    with col2:
        user_budget_custo = st.slider(
            "Qual seu orçamento máximo para entradas (R$)?",
            min_value=0, max_value=200, value=50, step=5
        )
    
    user_budget_min = user_budget_horas * 60
    
    # Botão para executar o algoritmo
    if st.button("🚀 Calcular Rota Otimizada (Heurística)"):
        
        # Chama a função do arquivo 'algoritmos.py'
        route_nodes, summary, log = alg.solve_budget_route_heuristic(
            all_nodes, 
            dist_matrix_full, 
            id_to_index,
            user_budget_min,
            user_budget_custo,
            start_node_id=1 # Começa no Jardim Botânico
        )
        
        st.divider()
        st.subheader("Resultados da Otimização")

        if not route_nodes:
            st.error("Não foi possível gerar uma rota. O orçamento é muito baixo até para o ponto de partida.")
        else:
            # --- Dashboard de Resultados (Spec 4.4) ---
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric(
                label="Custo Total Gasto",
                value=f"R$ {summary['custo_total_gasto']:.2f}",
                delta=f"R$ {summary['custo_max'] - summary['custo_total_gasto']:.2f} (sobra)"
            )
            kpi2.metric(
                label="Tempo Total Gasto",
                value=f"{summary['tempo_total_gasto']:.0f} min",
                delta=f"{summary['tempo_max'] - summary['tempo_total_gasto']:.0f} min (sobra)"
            )
            kpi3.metric(
                label="Score de Popularidade",
                value=f"{summary['score_popularidade']:.0f}"
            )
            
            st.subheader("Rota Sugerida")
            st.markdown(f"**Ordem de visita:** {summary['path_names']}")

            # Visualização no Mapa
            route_df = pd.DataFrame(route_nodes)
            st.map(route_df, latitude='latitude', longitude='longitude', size=50)

            # Log de execução
            with st.expander("Ver log de execução do algoritmo"):
                st.code("\n".join(log), language=None)


# =============================================================================
# ABA 3: EXPERIMENTOS TSP B&B (Spec 3.1, 4.3, 4.4)
# =============================================================================
with tab_tsp:
    st.header("Otimização de Rota Completa (Branch and Bound vs Branch and Cut)")
    
    st.info("""
    Esta aba compara duas abordagens para o Problema do Caixeiro Viajante (TSP):
    1.  **B&B Puro:** Um algoritmo Branch and Bound implementado manualmente em Python (`algoritmos.py`).
    2.  **PuLP (Branch & Cut):** Uma formulação de Programação Linear Inteira que usa um solver profissional (CBC/GLPK), que aplica **Branch and Bound + Cutting Planes** (`solver_pulp.py`).
    """)
    
    if st.button("Executar 4 Experimentos (B&B Puro vs PuLP)"):
        st.subheader("Resultados dos Experimentos")
        
        # Prepara os 4 subconjuntos
        exp_list = {
            "10 Primeiros": all_nodes[0:10],
            "10 Últimos": all_nodes[10:20],
            "10 Pares": [node for node in all_nodes if node['id'] % 2 == 0],
            "10 Ímpares": [node for node in all_nodes if node['id'] % 2 != 0]
        }
        
        results_data = []
        path_details = []
        
        # Roda os 4 experimentos
        with st.spinner("Calculando rotas ótimas com B&B Puro e PuLP... (Isso pode levar alguns segundos)"):
            for name, nodes in exp_list.items():
                
                # --- 1. Rodar B&B Puro ---
                result_bnb = alg.run_tsp_experiment(name, nodes)
                
                # --- 2. Rodar PuLP (Branch & Cut) ---
                result_pulp = pulp_solver.solve_tsp_with_pulp(name, nodes)
                
                # Junta os resultados
                if result_bnb and result_pulp:
                    results_data.append({
                        "Experimento": name,
                        "Dist (B&B)": result_bnb['cost'],
                        "Tempo (B&B)": result_bnb['time'],
                        "Nós (B&B)": result_bnb['nodes'],
                        "Podas (B&B)": result_bnb['pruned'],
                        "Dist (PuLP)": result_pulp['cost'],
                        "Tempo (PuLP)": result_pulp['time'],
                    })
                    path_details.append({
                        "Experimento": name,
                        "Caminho (B&B)": result_bnb['path_names'],
                        "Caminho (PuLP)": result_pulp['path_names']
                    })
                elif result_bnb:
                    st.warning(f"Solver PuLP falhou para o experimento '{name}'. Mostrando apenas B&B.")
                else:
                    st.error(f"Ambos os solvers falharam para '{name}'.")
        
        st.success("Experimentos concluídos!")
        
        # Tabela de Métricas (Spec 3.2 e 4.4)
        st.subheader("Métricas de Execução (Comparação)")
        metrics_df = pd.DataFrame(results_data)
        
        st.dataframe(metrics_df.set_index("Experimento").style.format({
            "Dist (B&B)": "{:.2f} km",
            "Tempo (B&B)": "{:.4f} s",
            "Dist (PuLP)": "{:.2f} km",
            "Tempo (PuLP)": "{:.4f} s",
        }))
        
        # Compara as distâncias ótimas
        if not metrics_df.empty:
            dist_check = np.allclose(metrics_df['Dist (B&B)'], metrics_df['Dist (PuLP)'])
            if dist_check:
                st.success("✅ Verificado: Ambos os métodos encontraram a mesma distância ótima!")
            else:
                st.error("❌ Atenção: Os métodos encontraram distâncias ótimas diferentes. Verifique a implementação.")

        # Detalhes dos Caminhos
        st.subheader("Rotas Ótimas Encontradas")
        path_df = pd.DataFrame(path_details).set_index("Experimento")
        st.table(path_df)