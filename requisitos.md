# Escopo da análise

Dados públicos anonimizados de transações: https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv . Consulta22/11/2026.

- Inspecionar Time, Amount, Class eV1–V28, distribuição das classes e duplicatas.
- Transformar valor com log1p e ajustar scaler somente no treino, emPipeline.
- Separar treino/validação/teste estratificados.
- Comparar regressão logística, RandomForest eXGBoost com pesos de classe.
- Avaliar recall, precisão, F1, ROC eprecisão-recall, com limiares0,5 e0,3.
- Selecionar pela validação e manter teste reservado.
- Explicar previsões comSHAP e explicitar limites da amostra.
- Publicar notebook executado, figuras, métricas e instruções reproduzíveis, sem redistribuir oCSV.

Decisões: deduplicação antes do split; busca pequena de hiperparâmetros; sem usar acurácia como critério único; sem aprovação para uso financeiro real.
