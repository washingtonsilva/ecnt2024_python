# Importa os módulos necessários
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import kurtosis, skew
from pathlib import Path

# Define o caminho relativo para o arquivo Excel
path_to_capm = Path("data/raw/capm.xls")

# Importa a planilha Excel
dados_capm = pd.read_excel(path_to_capm, engine='xlrd')

# Exibe a estrutura dos dados
print("Estrutura dos dados:")
print(dados_capm.info())

# Exibe as 5 primeiras e últimas linhas dos dados
print("Primeiras e últimas linhas dos dados:")
print(dados_capm)

# Cálculo dos retornos compostos continuamente (logarítmicos) e adiciona os retornos ao DataFrame
dados_capm['ret_sp500'] = np.log(dados_capm['SANDP']).diff() * 100
dados_capm['ret_ford'] = np.log(dados_capm['FORD']).diff() * 100
dados_capm['ret_ge'] = np.log(dados_capm['GE']).diff() * 100
dados_capm['ret_microsoft'] = np.log(dados_capm['MICROSOFT']).diff() * 100
dados_capm['ret_oracle'] = np.log(dados_capm['ORACLE']).diff() * 100

# Verifica a estrutura dos dados após o cálculo dos retornos
print("Estrutura dos dados após o cálculo dos retornos:")
print(dados_capm.info())

# Ajusta a série anual de rendimentos para base mensal
dados_capm['ustb3m'] = dados_capm['USTB3M'] / 12

# Cálculo dos retornos excedentes
dados_capm['retexc_sp500'] = dados_capm['ret_sp500'] - dados_capm['ustb3m']
dados_capm['retexc_ford'] = dados_capm['ret_ford'] - dados_capm['ustb3m']
dados_capm['retexc_ge'] = dados_capm['ret_ge'] - dados_capm['ustb3m']
dados_capm['retexc_microsoft'] = dados_capm['ret_microsoft'] - dados_capm['ustb3m']
dados_capm['retexc_oracle'] = dados_capm['ret_oracle'] - dados_capm['ustb3m']

# Exibe a estrutura dos dados após o cálculo dos retornos excedentes
print("Estrutura dos dados após o cálculo dos retornos excedentes:")
print(dados_capm.info())

# Exibe as 5 primeiras e últimas linhas dos dados
print("Primeiras e últimas linhas dos dados após o cálculo dos retornos excedentes:")
print(dados_capm)

# Estatísticas descritivas dos retornos excedentes da Ford
ford_stats = dados_capm['retexc_ford'].describe()
ford_skew = skew(dados_capm['retexc_ford'].dropna())
ford_kurt = kurtosis(dados_capm['retexc_ford'].dropna(), fisher=False)

ford_summary = pd.Series({
    'media': ford_stats['mean'],
    'mediana': ford_stats['50%'],
    'desvio_padrao': ford_stats['std'],
    'minimo': ford_stats['min'],
    'maximo': ford_stats['max'],
    'curtose': ford_kurt,
    'assimetria': ford_skew
})

print("Estatísticas descritivas dos retornos excedentes da Ford:")
print(ford_summary)

# Estatísticas descritivas dos retornos excedentes do S&P 500
sp500_stats = dados_capm['retexc_sp500'].describe()
sp500_skew = skew(dados_capm['retexc_sp500'].dropna())
sp500_kurt = kurtosis(dados_capm['retexc_sp500'].dropna(), fisher=False)

sp500_summary = pd.Series({
    'media': sp500_stats['mean'],
    'mediana': sp500_stats['50%'],
    'desvio_padrao': sp500_stats['std'],
    'minimo': sp500_stats['min'],
    'maximo': sp500_stats['max'],
    'curtose': sp500_kurt,
    'assimetria': sp500_skew
})

print("Estatísticas descritivas dos retornos excedentes do S&P 500:")
print(sp500_summary)

# Gráfico de dispersão dos retornos excedentes da Ford vs. S&P 500
sns.lmplot(x='retexc_sp500', y='retexc_ford', data=dados_capm, line_kws={'color': 'red'}, ci=None)
plt.title('Retornos Excedentes da Ford vs. Retornos Excedentes do S&P 500')
plt.xlabel('Retornos Excedentes do S&P 500 (%)')
plt.ylabel('Retornos Excedentes da Ford (%)')
plt.show()

# Preparação dos dados para estimação do CAPM
X = dados_capm['retexc_sp500'].dropna()
y = dados_capm['retexc_ford'].dropna()

# Adiciona uma constante para o termo de interceptação
X = sm.add_constant(X)

# Estimação do modelo CAPM
capm_model = sm.OLS(y, X).fit()

# Exibe os resultados da estimação
print("Resultados da estimação do modelo CAPM:")
print(capm_model.summary())

# Intervalos de confiança para os parâmetros estimados
print("Intervalos de confiança para os parâmetros estimados:")
print(capm_model.conf_int())

# Definindo a semente para reprodutibilidade
np.random.seed(1234)

# Simulação dos retornos excedentes da Ford com base no modelo estimado
retexc_ford_sim = capm_model.params['const'] + capm_model.params['retexc_sp500'] * dados_capm['retexc_sp500'] + np.random.normal(0, capm_model.resid.std(), len(dados_capm))

# Adiciona os retornos simulados ao DataFrame
dados_capm['retexc_ford_sim'] = retexc_ford_sim

# Gráfico comparando as densidades dos retornos simulados e observados
sns.kdeplot(dados_capm['retexc_ford_sim'].dropna(), color='red', label='Densidade dos Retornos Simulados')
sns.kdeplot(dados_capm['retexc_ford'].dropna(), color='blue', label='Densidade dos Retornos Observados')
plt.legend()
plt.show()