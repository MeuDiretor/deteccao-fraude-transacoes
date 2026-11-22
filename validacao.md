# Validação — 22/11/2026

- Base oficial: 150.828.752 bytes; SHA256 `76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
- Execução integral: `python build_notebook.py --execute`, saída0.
- Notebook: seis células de código executadas consecutivamente1–6; zero outputs de erro; formato validado com nbformat. Resumo em Markdown acrescentado a partir das métricas executadas.
- Reconciliados esquema/nulos/Amount não negativo/classes, duplicatas, conjuntos disjuntos e métricas de recall/precisão recalculadas pela matriz.
- Três modelos treinados, GridSearchCV com três folds, dois limiares comparados, métricas JSON, ROC/PR, matriz e SHAP local/agregado gerados. Sem células falsas ou resultados inventados.
- Quatro figuras renderizadas conferidas visualmente em captura conjunta; títulos, eixos, rótulos e números legíveis. Saídas no notebook preservadas.
- Falha inicial de chamada do kernel (argumento duplicado `kernel_cmd`) corrigida usando KernelManager com executável do ambiente; execução completa posterior aprovada.
- Não se afirma que maximizar recall produz modelo operacional aceitável: 3.187 falsos positivos e precisão2,54% da seleção experimental são limitação central, explicitada no README/notebook.

Este arquivo registra validação técnica local. Revisão de apresentação em22/11/2026 preservou todas as células de código,contagens de execução,saídas,métricas e figuras;alterou texto de contexto e suprimiu somente o caminho absoluto do ambiente em um aviso de dependência, mantendo o aviso e todos os resultados.
