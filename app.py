import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import algoritmos as alg
import solver_pulp as pulp_solver 

# --- Configuração da Página ---
st.set_page_config(
    page_title="Otimizador de Rotas CWB",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constante (para o README) ---
KAGGLE_URL = "https.kaggle.com/datasets/mathiasart/turismo-em-curitiba"


# =============================================================================
# CSS 10/10 - O GÊNIO (COM A CORREÇÃO DE SINTAXE)
# =============================================================================
# Cor de destaque principal, tirada do PDF da sua aula
PDF_YELLOW = "#FFC300" 

# CORREÇÃO: Todos os colchetes {} do CSS foram dobrados para {{ }}
# para funcionar dentro da f-string do Python.
CSS = f"""
<style>
/* --- Importar Fonte do Google --- */
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');

/* --- Fonte Global --- */
html, body, [class*="st-"], .st-emotion-cache-1n4a2v9, .st-emotion-cache-1pxazr4 {{
    font-family: 'Montserrat', sans-serif;
}}

/* --- Tema Dark Principal --- */
[data-testid="stAppViewContainer"] {{
    background-color: #111111; /* Fundo preto */
    color: #FAFAFA;
}}

/* --- Título Principal (Este usa a variável, então fica com colchete simples) --- */
[data-testid="stAppViewContainer"] > section > div > div > div > h1 {{
    color: {PDF_YELLOW};
    font-weight: 700;
}}

/* --- Divisores "Rainbow" --- */
hr {{
    background: linear-gradient(to right, {PDF_YELLOW}, #F5A623, {PDF_YELLOW});
    height: 2px !important;
    border: none;
}}

/* --- Barra Lateral (Sidebar) --- */
[data-testid="stSidebar"] {{
    background-color: #191919;
    border-right: 1px solid #333333;
}}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {PDF_YELLOW};
}}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
    color: #AFAFAF;
}}

/* --- Botões da Barra Lateral --- */
[data-testid="stRadio"] label {{
    padding: 0.5rem 0.75rem;
    border-radius: 5px;
    transition: all 0.3s ease;
    color: #FAFAFA;
}}
[data-testid="stRadio"] label:hover {{
    background-color: #333333;
}}
/* Botão de Rádio Selecionado */
[data-testid="stRadio"] [aria-checked="true"] {{
    background-color: rgba(255, 195, 0, 0.1);
    border-left: 3px solid {PDF_YELLOW};
    color: {PDF_YELLOW};
    font-weight: 600;
}}

/* --- Botão de Ação Principal --- */
.stButton > button {{
    background: linear-gradient(to right, {PDF_YELLOW}, #F5A623);
    color: #111111;
    font-weight: 700;
    border: none;
    border-radius: 5px;
    padding: 0.75rem 1.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255, 195, 0, 0.2);
}}
.stButton > button:hover {{
    background: linear-gradient(to right, #FFFFFF, #FAFAFA);
    color: #111111;
    transform: scale(1.02);
    box-shadow: 0 4px 20px rgba(255, 255, 255, 0.3);
}}

/* --- "Cards" de Container (Onde o st.container(border=True) é usado) --- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: #222222;
    border: 1px solid #333333;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    padding: 1.5rem;
}}

/* --- Headers dos Cards --- */
[data-testid="stVerticalBlockBorderWrapper"] h3 {{
    color: {PDF_YELLOW};
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
[data-testid="stVerticalBlockBorderWrapper"] h4, [data-testid="stVerticalBlockBorderWrapper"] h5 {{
    color: #FAFAFA;
    font-weight: 600;
}}

/* --- Cartões de KPI (Métricas) --- */
[data-testid="stMetric"] {{
    background-color: #1E1E1E;
    border: 1px solid #333333;
    border-left: 6px solid {PDF_YELLOW};
    border-radius: 8px;
    padding: 1rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}
/* Label (ex: "Economia Gerada") */
[data-testid="stMetricLabel"] {{
    color: #AFAFAF;
    font-size: 0.9rem;
}}
/* Valor (ex: "R$ 100,00") */
[data-testid="stMetricValue"] {{
    color: #FFFFFF;
    font-size: 2.2rem;
    font-weight: 700;
}}
/* Delta (ex: "-15%") */
[data-testid="stMetricDelta"] {{
    font-size: 1.1rem;
    font-weight: 600;
}}
/* Delta Negativo (Economia) */
[data-testid="stMetricDelta"] .st-emotion-cache-18bqsvl {{
    color: #00C49A; /* Verde para economia */
}}

/* --- Expanders --- */
[data-testid="stExpander"] summary {{
    font-weight: 600;
    color: {PDF_YELLOW};
    font-size: 1.1rem;
}}

/* --- Info Box --- */
[data-testid="stInfo"] {{
    background-color: rgba(255, 195, 0, 0.05);
    border: 1px solid {PDF_YELLOW};
}}

/* --- Success Box --- */
[data-testid="stSuccess"] {{
    background-color: rgba(0, 196, 154, 0.1);
    border: 1px solid #00C49A;
}}

</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --- Funções Auxiliares de Layout ---
def format_path_as_list(path_names_str):
    """Formata a string de rota 'A -> B -> C' em uma lista numerada em Markdown."""
    points = path_names_str.split(" -> ")
    md_list = ""
    for i, point in enumerate(points):
        md_list += f"**{i+1}.** {point}\n"
    return md_list

# --- Carregamento de Dados (Cache) ---
@st.cache_data
def load_data_cached():
    """ Carrega, limpa e prepara os dados, armazenando em cache."""
    df, all_nodes, id_map, dist_matrix = alg.load_data()
    if df is None:
        st.error(f"Erro fatal ao carregar o arquivo '{alg.CSV_FILE}'. Verifique se o arquivo está na pasta.")
        st.stop()
    
    jardim_botanico_node = next((node for node in all_nodes if node['id'] == 1), None)
    
    if jardim_botanico_node is None:
        st.error("Erro fatal: Jardim Botânico (ID 1) não encontrado no CSV.")
        st.stop()
        
    df_sem_jb = df[df['id'] != 1].copy()
    
    return df, all_nodes, id_map, dist_matrix, jardim_botanico_node, df_sem_jb

# Carrega os dados
df, all_nodes, id_to_index, dist_matrix_full, JARDIM_BOTANICO, df_sem_jb = load_data_cached()


# =============================================================================
# PÁGINA 1: ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================
def render_eda_page():
    st.header("📊 Análise Exploratória dos Pontos Turísticos", divider='rainbow')
    
    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total de Pontos", f"{len(df)}")
        col2.metric("Custo Médio Entrada", f"R$ {df['custo_entrada'].mean():.2f}")
        col3.metric("Tempo Médio Visita", f"{df['tempo_visita_min'].mean():.0f} min")
        col4.metric("Avaliação Média", f"{df['avaliacao'].mean():.1f} ⭐")
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.subheader("Distribuição de Custos de Entrada")
            chart_custo = alt.Chart(df).mark_bar().encode(
                x=alt.X('custo_entrada:Q', bin=True, title='Custo da Entrada (R$)'),
                y=alt.Y('count()', title='Contagem de Locais'),
                tooltip=['custo_entrada', 'count()']
            ).interactive()
            st.altair_chart(chart_custo, use_container_width=True)
        
        with st.container(border=True):
            st.subheader("Popularidade vs. Avaliação")
            chart_pop_aval = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('avaliacao:Q', title='Avaliação (0-5)'),
                y=alt.Y('popularidade:Q', title='Popularidade (0-100)'),
                color='categoria',
                tooltip=['nome', 'avaliacao', 'popularidade', 'categoria']
            ).interactive()
            st.altair_chart(chart_pop_aval, use_container_width=True)

    with c2:
        with st.container(border=True):
            st.subheader("Tempo de Visita por Categoria")
            chart_tempo_cat = alt.Chart(df).mark_boxplot().encode(
                x=alt.X('categoria:N', title='Categoria'),
                y=alt.Y('tempo_visita_min:Q', title='Tempo de Visita (min)'),
                tooltip=['categoria', 'tempo_visita_min']
            ).interactive()
            st.altair_chart(chart_tempo_cat, use_container_width=True)

    with st.expander("Ver Tabela de Dados Completa", expanded=False):
        with st.container(border=True):
            st.dataframe(df, height=300)

# =============================================================================
# PÁGINA 2: ROTA POR ORÇAMENTO (HEURÍSTICA)
# =============================================================================
def render_budget_page(user_budget_min, user_budget_custo, btn_calc_budget):
    st.header("💰 Planejador de Rota por Orçamento", divider='rainbow')
    st.markdown("Defina seu orçamento de tempo e custo na barra lateral para encontrar a melhor rota (maximizando popularidade), **partindo do Jardim Botânico**.")

    if btn_calc_budget:
        route_nodes, summary, log = alg.solve_budget_route_heuristic(
            all_nodes, 
            dist_matrix_full, 
            id_to_index,
            user_budget_min,
            user_budget_custo,
            start_node_id=JARDIM_BOTANICO['id']
        )
        
        st.subheader("Resultados da Otimização")

        if not route_nodes:
            st.error("Não foi possível gerar uma rota. O orçamento é muito baixo até para o ponto de partida.")
        else:
            with st.container(border=True):
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Custo Total Gasto", f"R$ {summary['custo_total_gasto']:.2f}", f"R$ {summary['custo_max'] - summary['custo_total_gasto']:.2f} (sobra)")
                kpi2.metric("Tempo Total Gasto", f"{summary['tempo_total_gasto']:.0f} min", f"{summary['tempo_max'] - summary['tempo_total_gasto']:.0f} min (sobra)")
                kpi3.metric("Score de Popularidade", f"{summary['score_popularidade']:.0f}")
            
            with st.container(border=True):
                st.subheader("Rota Sugerida")
                st.markdown(f"**Ordem de visita:** {summary['path_names']}")
                route_df = pd.DataFrame(route_nodes)
                st.map(route_df, latitude='latitude', longitude='longitude', size=50)

            with st.expander("Ver log de execução da heurística", expanded=False):
                st.code("\n".join(log), language=None)
    else:
        st.info("Ajuste os parâmetros na barra lateral e clique em 'Calcular Rota por Orçamento'.")

# =============================================================================
# PÁGINA 3: OTIMIZADOR DE ROTA (TSP) - LAYOUT 10/10
# =============================================================================
def render_tsp_page(selected_node_names, cost_per_km, cost_per_hour, avg_speed_kmh, btn_calc_tsp):
    st.header("🚚 Otimizador de Rota (TSP) com Análise de Budget", divider='rainbow')
    st.markdown("Selecione na barra lateral os pontos que deseja visitar. O sistema calculará a rota mais curta **(partindo e voltando ao Jardim Botânico)** e o impacto financeiro dessa otimização.")

    if btn_calc_tsp:
        if len(selected_node_names) < 2:
            st.error("Por favor, selecione pelo menos 2 pontos para visitar (além do Jardim Botânico).")
            return

        selected_nodes_data = [node for node in all_nodes if node['nome'] in selected_node_names]
        nodes_for_solver = [JARDIM_BOTANICO] + selected_nodes_data
        experiment_name = f"Rota de {len(nodes_for_solver)} pontos"
        
        with st.spinner(f"Calculando rotas ótimas para '{experiment_name}'... (Isso pode levar alguns segundos)"):
            result_bnb = alg.run_tsp_experiment(experiment_name, nodes_for_solver)
            result_pulp = pulp_solver.solve_tsp_with_pulp(experiment_name, nodes_for_solver)

        if not result_bnb or not result_pulp:
            st.error("Falha ao calcular a rota. Verifique o console para mais detalhes.")
            return

        st.success(f"Otimização concluída para {experiment_name}!")

        # --- Cálculos de Budget ---
        dist_atual = result_bnb['heuristic_cost']
        dist_otimizada = result_bnb['cost']
        time_atual_h = dist_atual / avg_speed_kmh
        time_otimizada_h = dist_otimizada / avg_speed_kmh
        cost_km_atual = dist_atual * cost_per_km
        cost_hora_atual = time_atual_h * cost_per_hour
        total_atual = cost_km_atual + cost_hora_atual
        cost_km_otimizado = dist_otimizada * cost_per_km
        cost_hora_otimizado = time_otimizada_h * cost_per_hour
        total_otimizado = cost_km_otimizado + cost_hora_otimizado
        economia_total = total_atual - total_otimizado
        economia_percent = (economia_total / total_atual) * 100 if total_atual > 0 else 0

        # --- SEÇÃO 1: RESULTADOS FINANCEIROS (KPIs "Heróis") ---
        st.subheader("📈 Análise de Impacto (Budget) - Requisito A1")
        with st.container(border=True):
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Custo (Cenário Atual/Heurística)", f"R$ {total_atual:.2f}")
            kpi2.metric("Custo (Cenário Otimizado/B&B)", f"R$ {total_otimizado:.2f}")
            kpi3.metric("Economia Gerada (Lucro)", f"R$ {economia_total:.2f}", f"{economia_percent:.1f}%", delta_color="inverse")

        st.header("⚙️ Dashboards de Otimização", divider='rainbow')

        # --- SEÇÃO 2: LAYOUT DE DASHBOARD 2:1 ---
        col_map, col_metrics = st.columns([2, 1]) # Coluna do mapa 2x mais larga

        with col_map:
            # --- CARD 2.1: MAPA E ROTA ---
            with st.container(border=True):
                st.subheader("🗺️ Rota Otimizada (B&B)")
                route_nodes_optimized = [nodes_for_solver[i] for i in result_bnb['path']]
                route_df_optimized = pd.DataFrame(route_nodes_optimized)
                st.map(route_df_optimized, latitude='latitude', longitude='longitude', size=50, height=400)
                
                st.subheader("Ordem de Visita")
                st.markdown(format_path_as_list(result_bnb['path_names']))

        with col_metrics:
            # --- CARD 2.2: MÉTRICAS DO B&B (REQUISITO 4.3) ---
            with st.container(border=True):
                st.subheader("📊 Métricas do Algoritmo (B&B)")
                kpi_b1, kpi_b2, kpi_b3 = st.columns(3)
                kpi_b1.metric("Nós", f"{result_bnb['nodes']:,}")
                kpi_b2.metric("Podas", f"{result_bnb['pruned']:,}")
                kpi_b3.metric("Tempo (s)", f"{result_bnb['time']:.4f}")
                st.divider()
                st.subheader("📈 Limites (Bounds)")
                kpi_l1, kpi_l2 = st.columns(2)
                kpi_l1.metric("Superior (Heurística)", f"{result_bnb['heuristic_cost']:.2f} km")
                kpi_l2.metric("Ótimo (B&B)", f"{result_bnb['cost']:.2f} km")
            
            # --- CARD 2.3: COMPARAÇÃO DE SOLVERS ---
            with st.container(border=True):
                st.subheader("⏱️ Validação (B&B vs PuLP)")
                data_perf = {
                    "Métrica": ["Distância (km)", "Tempo (s)"],
                    "B&B Puro (Python)": [f"{result_bnb['cost']:.2f}", f"{result_bnb['time']:.4f}"],
                    "PuLP (Branch & Cut)": [f"{result_pulp['cost']:.2f}", f"{result_pulp['time']:.4f}"]
                }
                st.dataframe(pd.DataFrame(data_perf).set_index('Métrica'), use_container_width=True)
                if np.allclose(result_bnb['cost'], result_pulp['cost']):
                    st.success("✅ Verificado: Soluções idênticas!")
                else:
                    st.error("❌ Atenção: Soluções divergentes.")
            
            with st.expander("Ver Detalhamento Financeiro (Tabela)"):
                data_budget = {
                    'Métrica': ['Distância Total (km)', 'Tempo de Viagem (horas)', 'Custo Deslocamento (R$)', 'Custo Mão de Obra (R$)', '**Custo Total (Budget)**'],
                    'Cenário Atual (Não Otimizado)': [f"{dist_atual:.2f} km", f"{time_atual_h:.2f} h", f"R$ {cost_km_atual:.2f}", f"R$ {cost_hora_atual:.2f}", f"**R$ {total_atual:.2f}**"],
                    'Cenário Otimizado (B&B)': [f"{dist_otimizada:.2f} km", f"{time_otimizada_h:.2f} h", f"R$ {cost_km_otimizado:.2f}", f"R$ {cost_hora_otimizado:.2f}", f"**R$ {total_otimizado:.2f}**"],
                    'Variação (Economia)': [f"{(dist_otimizada - dist_atual):.2f} km", f"{(time_otimizada_h - time_atual_h):.2f} h", f"- R$ {(cost_km_atual - cost_km_otimizado):.2f}", f"- R$ {(cost_hora_atual - cost_hora_otimizado):.2f}", f"**- R$ {economia_total:.2f}**"]
                }
                st.dataframe(pd.DataFrame(data_budget).set_index('Métrica'), use_container_width=True)

    else:
        st.info("Ajuste os parâmetros na barra lateral e clique em 'Otimizar Rota e Calcular Impacto'.")


# =============================================================================
# PÁGINA 4: ANÁLISE DE SENSIBILIDADE
# =============================================================================
def render_sensitivity_page(cost_per_hour_sens, avg_speed_kmh_sens):
    st.header("🔬 Análise de Sensibilidade (Requisito 5.2)", divider='rainbow')
    st.markdown("Esta análise avalia o impacto de um parâmetro (Custo por KM) no resultado financeiro final (Custo Total da Rota), mantendo a rota otimizada fixa.")

    nodes_for_solver = [JARDIM_BOTANICO] + [node for node in all_nodes if node['nome'] in df_sem_jb['nome'].head(5).tolist()]
    result_bnb = alg.run_tsp_experiment("Rota Fixa (Sensibilidade)", nodes_for_solver)
    
    if not result_bnb:
        st.error("Não foi possível calcular a rota base para a análise.")
        return
        
    dist_otimizada = result_bnb['cost']
    st.info(f"Rota base para análise: **{dist_otimizada:.2f} km** (Partindo do JB, visitando 5 pontos). Parâmetros de Custo/Velocidade definidos na barra lateral.")

    cost_per_km_range = np.linspace(1.0, 5.0, 20)
    results_data = []
    time_cost = (dist_otimizada / avg_speed_kmh_sens) * cost_per_hour_sens
    
    for cost_km in cost_per_km_range:
        distance_cost = dist_otimizada * cost_km
        total_cost = distance_cost + time_cost
        results_data.append({"Custo por KM (R$)": cost_km, "Custo Total da Rota (R$)": total_cost})

    with st.container(border=True):
        st.subheader("Impacto do Custo por KM no Custo Total da Rota")
        df_sens = pd.DataFrame(results_data)
        chart = alt.Chart(df_sens).mark_line(point=True, color=PDF_YELLOW).encode(
            x=alt.X('Custo por KM (R$)', title='Custo por KM (R$)'),
            y=alt.Y('Custo Total da Rota (R$)', title='Custo Total da Rota (R$)'),
            tooltip=['Custo por KM (R$)', 'Custo Total da Rota (R$)']
        ).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.markdown(f"**Análise:** O gráfico demonstra uma **relação linear direta** entre o custo variável por KM e o custo total. O custo fixo de mão de obra (calculado em **R$ {time_cost:.2f}** para esta rota) define o intercepto (ponto inicial) da curva.")

# =============================================================================
# PÁGINA 5: MODELAGEM MATEMÁTICA
# =============================================================================
def render_modeling_page():
    st.header("🧮 Modelagem Matemática e Cálculos", divider='rainbow')
    st.markdown("Esta página detalha as fórmulas algébricas e os modelos de otimização utilizados no projeto.")

    with st.container(border=True):
        st.subheader("1. Otimizador de Rota (TSP - PuLP)")
        st.markdown("Usamos a formulação de **Miller-Tucker-Zemlin (MTZ)** para a Programação Linear Inteira (PLI).")
        st.markdown("##### a. Variáveis de Decisão")
        st.latex(r"x_{ij} = \begin{cases} 1 & \text{se a rota vai do nó } i \text{ para o nó } j \\ 0 & \text{caso contrário} \end{cases}")
        st.latex(r"u_i = \text{Variável auxiliar inteira para a ordem da visita}")
        st.markdown("##### b. Função Objetivo (Minimizar a Distância Total)")
        st.latex(r"\text{Minimizar} \sum_{i=0}^{n-1} \sum_{j=0, i \neq j}^{n-1} c_{ij} \cdot x_{ij}")
        st.markdown("##### c. Restrições")
        st.latex(r"\sum_{j=0, i \neq j}^{n-1} x_{ij} = 1 \quad (\text{Sair de cada nó uma vez})")
        st.latex(r"\sum_{i=0, i \neq j}^{n-1} x_{ij} = 1 \quad (\text{Entrar em cada nó uma vez})")
        st.latex(r"u_i - u_j + n \cdot x_{ij} \le n - 1 \quad (\text{Eliminação de Sub-rotas MTZ})")

    with st.container(border=True):
        st.subheader("2. Cálculo de Distância (Haversine)")
        st.latex(r"d = 2r \arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2 - \phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2 - \lambda_1}{2}\right)}\right)")

    with st.container(border=True):
        st.subheader("3. Rota por Orçamento (Heurística Gulosa)")
        st.markdown("Função de Score (Maximização):")
        st.latex(r"\text{Score} = \frac{\text{Popularidade}}{\text{Tempo de Deslocamento} + \text{Tempo de Visita}}")

    with st.container(border=True):
        st.subheader("4. Análise de Impacto (Budget)")
        st.latex(r"C_{\text{total}} = (D_{\text{total}} \times C_{\text{km}}) + \left(\frac{D_{\text{total}}}{V_{\text{media}}}\right) \times C_{\text{hora}}")

# =============================================================================
# PÁGINA 6: SOBRE O PROJETO
# =============================================================================
def render_about_page():
    st.header("ℹ️ Sobre o Projeto", divider='rainbow')
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://i.imgur.com/n0a7P8n.png", width=150) # Logo CWB
    with col2:
        st.markdown(f"""
        Este aplicativo é um projeto acadêmico para a disciplina de **Pesquisa Operacional** do Prof. Tiago Batista Pedra.
        O objetivo é aplicar conceitos de otimização para resolver problemas logísticos no contexto do turismo em Curitiba.
        """)
        
    st.markdown(f"""
    ### Fonte de Dados (Requisito 1.1)
    * **Fonte:** Kaggle
    * **Dataset:** Turismo em Curitiba (Pontos Turísticos)
    * **Link:** [{KAGGLE_URL}]({KAGGLE_URL})
    
    ### Métodos de Otimização Utilizados:
    1.  **Heurística Gulosa (Vizinho Mais Próximo):** Usada para gerar o "Cenário Atual" (não-otimizado) e como limite superior inicial para o B&B.
    2.  **Branch and Bound (B&B) Puro:** Algoritmo exato implementado manualmente em Python para encontrar a solução ótima do TSP.
    3.  **PuLP (Branch & Cut):** Formulação de PLI que usa um solver profissional para validar a solução ótima.
    """)

# =============================================================================
# LÓGICA PRINCIPAL (MAIN)
# =============================================================================

# --- Título Principal ---
st.title("🗺️ Otimizador de Rotas Turísticas CWB")
st.caption(f"Projeto de Pesquisa Operacional - Prof. Tiago Batista Pedra  |  Versão: 10/10")

# --- Barra Lateral (Sidebar) ---
st.sidebar.title("🗺️ MENU")

page_options = {
    "🏠 Análise Exploratória (EDA)": render_eda_page,
    "💰 Rota por Orçamento (Heurística)": render_budget_page,
    "🚚 Otimizador de Rota (TSP)": render_tsp_page,
    "🔬 Análise de Sensibilidade": render_sensitivity_page,
    "🧮 Modelagem Matemática": render_modeling_page,
    "ℹ️ Sobre o Projeto": render_about_page
}

page_selection = st.sidebar.radio("Escolha uma funcionalidade:", list(page_options.keys()))
st.sidebar.divider()
st.sidebar.caption(f"Fonte: [{KAGGLE_URL.split('/')[-1]}]({KAGGLE_URL})")


# --- Renderização Condicional de Controles ---
if page_selection == "💰 Rota por Orçamento (Heurística)":
    st.sidebar.header("Defina seu Orçamento")
    st.sidebar.info("**Ponto de Partida Fixo:**\nJardim Botânico")
    user_budget_horas = st.sidebar.slider("Horas disponíveis?", 1.0, 24.0, 8.0, 0.5)
    user_budget_custo = st.sidebar.slider("Orçamento para entradas (R$)?", 0, 200, 50, 5)
    user_budget_min = user_budget_horas * 60
    btn_calc_budget = st.sidebar.button("🚀 Calcular Rota por Orçamento", use_container_width=True)
    
    render_budget_page(user_budget_min, user_budget_custo, btn_calc_budget)

elif page_selection == "🚚 Otimizador de Rota (TSP)":
    st.sidebar.header("Defina sua Rota Otimizada")
    st.sidebar.info("**Ponto de Partida e Chegada Fixo:**\nJardim Botânico")
    selected_node_names = st.sidebar.multiselect("Selecione os pontos para visitar (2 a 9):", 
                                                 options=df_sem_jb['nome'], 
                                                 default=df_sem_jb['nome'].head(5).tolist(), 
                                                 max_selections=9)
    st.sidebar.divider()
    st.sidebar.subheader("Definição de Custos Variáveis")
    cost_per_km = st.sidebar.number_input("Custo por KM (R$)", 0.1, 10.0, 2.50, 0.1, key="tsp_km")
    cost_per_hour = st.sidebar.number_input("Custo por Hora (Guia) (R$)", 1.0, 200.0, 30.0, 1.0, key="tsp_hr")
    avg_speed_kmh = st.sidebar.number_input("Velocidade Média (km/h)", 1.0, 80.0, float(alg.AVG_SPEED_KMH), 1.0, key="tsp_spd")
    btn_calc_tsp = st.sidebar.button("📊 Otimizar Rota e Calcular Impacto", use_container_width=True)
    
    render_tsp_page(selected_node_names, cost_per_km, cost_per_hour, avg_speed_kmh, btn_calc_tsp)

elif page_selection == "🔬 Análise de Sensibilidade":
    st.sidebar.subheader("Parâmetros (Sensibilidade)")
    st.sidebar.info("Ajuste os parâmetros de custo fixo para ver o impacto no gráfico.")
    cost_per_hour_sens = st.sidebar.number_input("Custo por Hora (R$)", 1.0, 200.0, 30.0, 1.0, key="sens_hr")
    avg_speed_kmh_sens = st.sidebar.number_input("Velocidade Média (km/h)", 1.0, 80.0, float(alg.AVG_SPEED_KMH), 1.0, key="sens_spd")
    
    render_sensitivity_page(cost_per_hour_sens, avg_speed_kmh_sens)

else:
    # Renderiza as páginas que não têm controles na sidebar
    page_options[page_selection]()