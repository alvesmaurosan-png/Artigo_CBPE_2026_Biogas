# Dados

## Arquivos publicados

- `raw/fleet_demand_sp_thesis.csv`: conjunto de origem usado na conversão.
- `raw/fleet_demand_sp_synthetic.csv`: alternativa sintética preservada para comparação.
- `processed/fleet_demand_sp.csv`: entrada canônica do artigo, com 8.760 registros.

O arquivo processado possui as colunas `hour`, `demand_kw` e `pv_factor`.
`hour` é um índice anual de zero a 8.759, a demanda é não negativa e o fator PV
é normalizado entre zero e um. A transformação histórica está preservada em
`data/convert_thesis_dataset.py`; a interface pública valida o produto antes de
cada execução.

O SHA-256 é calculado e gravado em `results/paper/manifest.json`. Isso evita que
uma alteração silenciosa dos dados seja confundida com variação do algoritmo.

Licença: CC BY 4.0. A atribuição deve incluir o artigo CBPE e a release deste
repositório. Os autores devem confirmar no registro da release a origem primária
e os responsáveis pela coleta antes da publicação definitiva.

