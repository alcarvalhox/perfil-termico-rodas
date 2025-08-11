import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from io import BytesIO

# Configuração da página e título
st.set_page_config(page_title="Análise de Perfil Térmico de Rodas", layout="wide")
st.title("🔍 Análise de Perfil Térmico de Rodas")
st.write("✅ App iniciado com sucesso! Faça o upload de um arquivo para começar.")

# Parâmetro de corte para a classificação
cut_off = 0.64

# Nome do arquivo do modelo
model_filename = 'modelo_p_t_rod_3_smt'

# Função para carregar o modelo em cache
@st.cache_resource
def load_model(filename):
    """Carrega o modelo de predição a partir de um arquivo local."""
    try:
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error(f"❌ Erro: O arquivo do modelo '{filename}' não foi encontrado na pasta do projeto.")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")
        return None

# Carregamento do arquivo de entrada pelo usuário
uploaded_file = st.file_uploader("📤 Faça o upload do arquivo Excel para análise", type=["xlsx"])

# Se o arquivo foi carregado, exibe o botão de análise
if uploaded_file is not None:
    st.success("📁 Arquivo carregado com sucesso!")
    st.info("Clique em 'Fazer a Análise' para processar os dados.")

    # Botão para iniciar a análise
    if st.button("Fazer a Análise"):
        try:
            # Lendo o arquivo Excel
            bd1 = pd.read_excel(uploaded_file, engine='openpyxl')
            array = bd1.values
            # O código original usava X = array[:,0:36]
            # Assumindo que as 36 primeiras colunas são as features
            X = array[:, 0:36]

            # Carregando o modelo
            model = load_model(modelo_p_t_rod_3_smt)
            if model is None:
                st.stop()

            # Realizando as predições
            preds = model.predict(X)
            preds_prob = model.predict_proba(X)
            
            # Aplicando o cut-off para a classificação binária
            results = np.where(preds_prob[:, 1] > cut_off, 1, 0)
            
            # Criando o DataFrame de predição
            predicao = pd.DataFrame(results, columns=['Resultado'])
            predicao['Resultado'] = predicao['Resultado'].apply(lambda x: "Verdadeiro" if x == 1 else "Falso")
            
            # Criando o DataFrame de probabilidades
            proba = pd.DataFrame(preds_prob, columns=['Falso(%)', 'Verdadeiro(%)'])
            proba['Falso(%)'] *= 100
            proba['Verdadeiro(%)'] *= 100
            
            # Concatenando o DataFrame original com os resultados
            teste = pd.DataFrame(X)
            df = pd.concat([teste, predicao, proba], axis=1)

            st.subheader("📊 Relatório Gerado")
            st.dataframe(df)

            # Geração e exibição de gráficos (exemplo)
            st.subheader("📈 Análises Gráficas")
            
            # Gráfico de barras da distribuição dos resultados
            fig_resultados = px.histogram(df, x='Resultado', color='Resultado',
                                        title='Distribuição dos Resultados')
            st.plotly_chart(fig_resultados)

            # Gráfico de histograma da probabilidade de "Verdadeiro" com linha de corte
            fig_prob_verdadeiro = px.histogram(df, x='Verdadeiro(%)',
                                               title='Distribuição da Probabilidade de "Verdadeiro"',
                                               marginal='box',
                                               color_discrete_sequence=['#1f77b4'])
            fig_prob_verdadeiro.add_vline(x=cut_off * 100, line_dash="dash", line_color="red", 
                                          annotation_text=f"Cut-off: {cut_off*100:.2f}%", 
                                          annotation_position="top right")
            st.plotly_chart(fig_prob_verdadeiro)

            # Função para converter o DataFrame em Excel para download
            def convert_df_to_excel(df):
                output = BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, index=False)
                return output.getvalue()

            excel_data = convert_df_to_excel(df)

            st.download_button(
                label="📥 Baixar Relatório em Excel",
                data=excel_data,
                file_name="relatorio_analise_rodas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.success("✅ Análise concluída e relatório gerado com sucesso!")

        except Exception as e:
            st.error(f"❌ Ocorreu um erro durante a análise: {e}")
            st.stop()
else:
    st.info("📁 Aguardando o upload do arquivo Excel para iniciar a análise.")
