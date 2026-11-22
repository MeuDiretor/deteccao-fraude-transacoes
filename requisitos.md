# Requisitos observados — DIO

Projeto: Detecção de Anomalias em Transações em Python.
Fonte: https://web.dio.me/lab/deteccao-de-anomalias-em-transacoes/learning/e38d0c1c-a7e9-4d74-b79c-9a5584578581?back=/track/formacao-python-for-data-analytics
Consulta: 22/11/2026. Base: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv . Sem repositório-base, notebook do zero.

- Explorar dataset público de cartão: Time, Amount, Class (1 fraude), V1–V28 anonimizadas por PCA; medir distribuição das classes.
- Aplicar `np.log1p` ao valor, StandardScaler e divisão estratificada.
- Comparar regressão logística, Random Forest com `class_weight='balanced'` e XGBoost; priorizar recall da fraude sem esconder precisão/F1.
- Relatórios de classificação, ROC e curva precisão-recall.
- Pipeline de transformação/modelo; comparar limiares0.5 e0.3; GridSearchCV com seleção por recall.
- Importância de variáveis e SHAP para explicar marcação de fraude.
- Ideias opcionais: under/oversampling, ampliar busca, variáveis temporais. Não substituir requisitos por estas extensões.
- Entregar notebook executável com saídas salvas, tabelas e curvas; README sobre problema, preparo, modelos, limiar, SHAP e diferenças próprias.
- Repositório público, minúsculas sem acento, na conta autorizada; link de repositório no formulário DIO.
- CSV não vai ao GitHub; todos arquivos citados devem existir e ter conteúdo. Conferir execução do início ao fim.

Decisões próprias: separar treino/validação/teste estratificados; selecionar modelo e limiar pela validação, preservando teste final; aplicar transformações dentro de Pipeline para evitar vazamento. Não usar acurácia como critério único. Resultados educacionais não aprovam uso financeiro real. Assistência de IA declarada no README.
