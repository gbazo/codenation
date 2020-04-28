#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[293]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[308]:


black_friday.head()


# In[309]:


black_friday.shape


# In[310]:


black_friday.isna().sum()


# In[311]:


black_friday.dtypes


# In[312]:


black_friday.describe()


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[13]:


def q1():
    return black_friday.shape
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[29]:


def q2():
    return len(black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')])
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[127]:


def q3():
    return black_friday['User_ID'].nunique()
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[126]:


def q4():
    return black_friday.dtypes.nunique()
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[301]:


def q5():
    return (black_friday.shape[0] - len(black_friday.dropna())) / black_friday.shape[0]
    pass


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[91]:


def q6():
    return max(black_friday.isnull().sum())
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[255]:


def q7():
    return black_friday['Product_Category_3'].mode().iloc[0]
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[285]:


def q8():
    return (((black_friday['Purchase'] - min(black_friday['Purchase'])) / (max(black_friday['Purchase']) - min(black_friday['Purchase']))).mean()).tolist() 
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[349]:


def q9():
    black_friday['padrao'] = (black_friday.Purchase - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    df2 = black_friday[black_friday['padrao'] >= -1]
    df2 = df2[df2['padrao'] <= 1]
    return df2.shape[0]
    pass


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[346]:


def q10():
    df = black_friday[['Product_Category_2','Product_Category_3']]
    df = df[df['Product_Category_2'].isna()]
    compare = df['Product_Category_2'].equals(df['Product_Category_3'])
    return compare
    pass

