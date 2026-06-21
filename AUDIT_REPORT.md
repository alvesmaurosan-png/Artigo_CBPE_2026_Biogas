# Relatório de auditoria científica

## Situação das alegações do manuscrito v02

| Alegação ou artefato | Fonte rastreada | Situação |
|---|---|---|
| Consumo anual de aproximadamente 2,27 GWh | `results/paper/source_data/master_summary.csv` | Compatível: 2.268.096 kWh |
| Solução base de 722 kW PV, 854 kWh BSV, 355 kW eletrólise, 358 kg H2 e 87 kW FC | mesmo arquivo, cenário `tariff_050_with_h2` | Compatível com o resultado arquivado |
| LCOE na faixa de 0,15-0,18 USD/kWh | múltiplos Pareto arquivados | Requer delimitar cenário; há valores abaixo de 0,15 e acima de 0,18 |
| Limiar estrutural de aproximadamente 468 kW | Pareto PV+BSV restrito | Não demonstrado como inviabilidade; 468,25 kW é o mínimo observado |
| H2 reduz mínimo para aproximadamente 351 kW | Pareto PV+BSV+H2, BSV <= 1500 kWh | Compatível, mas pertence a configuração tecnológica distinta |
| Figura 1 | análise de mínimos observados e futuro resultado anual | Renomeada para não afirmar limiar antes da prova formal |
| Figura 2 | `weekly_typical_dispatch.csv` | Regenerável |
| Figura 3 | `annual_vs_peak.csv` | Regenerável |
| Figura 4 e Tabela 2 | quatro arquivos Pareto canônicos | Regeneráveis |

## Falha histórica de classificação

Os casos de capacidade da rede entre 450 e 500 kW foram marcados como
inviáveis após `run_single_case.py` terminar com `UnicodeEncodeError` ao
imprimir um símbolo no console CP1252. Os resultados do solver haviam sido
produzidos. Falha de apresentação não é evidência de inviabilidade.

A nova camada usa estados separados: `OPTIMAL`, `FEASIBLE`, `INFEASIBLE`,
`TIME_LIMIT` e `EXECUTION_ERROR`. Um resultado científico válido tem prioridade
sobre falhas posteriores do processo invocador.

## Limitações ainda abertas

- O NSGA-II é uma busca heurística; ausência de indivíduo não prova que o
  espaço viável é vazio.
- A análise anual integrada deve convergir com gap documentado antes de a
  expressão “limiar estrutural” ser mantida no artigo.
- Os resultados arquivados foram produzidos no Windows com Python 3.13 e
  OR-Tools/SCIP; a release final deve registrar a repetição em ambiente limpo.

## Execução anual de auditoria em 21/06/2026

Com limite de 120 segundos por configuração, o modelo anual integrado encontrou
incumbente de 474,81 kW nos dois cenários. Para PV+BSV, o melhor limite inferior
foi 379,90 kW (gap de 20,0%). Para PV+BSV+H2, o limite permaneceu em 0 kW (gap de
100%). Ambos os casos são `TIME_LIMIT`, com incumbente preservado, e não
autorizam conclusão sobre um limiar estrutural. O arquivo rastreado é
`results/paper/tables/feasibility_annual.json`.
