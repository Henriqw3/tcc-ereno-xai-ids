# TCC Ereno XAI-IDS

Este repositório contém o código-fonte e os arquivos relacionados ao projeto de Trabalho de Conclusão de Curso (TCC) sobre o aprimoramento de um IDS de Subestações Elétrica.

## Descrição

O objetivo deste projeto é desenvolver um sistema de detecção de intrusão (IDS) explicável, utilizando o framework Ereno. usando o framework Ereno. As técnicas de aprimoramento tiveram como base enriquecimento de dados temporais, extração, normalização de features e Explainable AI (XAI).

## Funcionalidades

- **Detecção de Intrusões**: Com o uso dos dados do framework Ereno UI e utilizando técnicas de aprendizado de máquina, o sistema é capaz de identificar padrões anormais que possam indicar uma tentativa de intrusão.
- **Explicabilidade**: Com o uso do framework SHAP, o sistema fornece explicações interpretáveis sobre as decisões tomadas, auxiliando na compreensão e na confiança no processo de detecção.
- **Enriquecimento Temporal**: Realiza uma extração de features por janelas de tempo segmentada e não sobreposta, onde cada evento terá informações da sua janela de tempo, como média da janela em que o evento faz parte, variância, desvio padrão, kurtosis e skewness para verificar a assimetria e dispersão dos tempos contidos em cada janela.
- **Pré processamento de dados**: Com o uso de MinMax Scaler, OneHotEncoder e LabelEncoder os dados numéricos e categóricos são normalizados.
- **Trigger Automático**: O sistema inclui um mecanismo de trigger automático que monitora uma pasta específica e executa a conversão de novos arquivos assim que eles são adicionados pelo Framework que exporta em formato .arff

## Requisitos

- Linguagem: Python 3.x


| Bibliotecas Python | Version |
|---------------------|---------|
| catboost            | 1.2.3   |
| ipython             | 8.20.0  |
| matplotlib          | 3.8.2   |
| numpy               | 1.26.2  |
| pandas              | 2.1.3   |
| scikit-learn        | 1.3.2   |
| scipy               | 1.11.4  |
| seaborn             | 0.13.0  |
| shap                | 0.44.0  |
| watchdog            | 4.0.0   |
| xgboost             | 2.0.2   |

## Instalação

1. Clone este repositório em sua máquina local:
```
git clone https://github.com/seu-usuario/tcc-ereno-xai-ids.git
```
2. Acesse o [ERENO UI FRAMEWORK](URL) e use a ferramenta para simular dados de redes de subestações elétricas.

3. Inicie o script trigger que monitorará a pasta `inputs_ereno_ui` em busca de novos arquivos `.arff` ou `.csv`. Certifique-se de adicionar seus arquivos `.arff` à pasta `inputs_ereno_ui` para que o trigger automático os converta corretamente.

4. Configure os caminhos de pasta que será usado pelo Main, após isso basta executar o Main para os arquivos csv nas devidas pastas.

## Contribuição

Todas aprimorações e contribuições são bem vindas :)<br>
Se você encontrar algum problema ou tiver sugestões de melhorias, sinta-se à vontade para abrir uma nova issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
