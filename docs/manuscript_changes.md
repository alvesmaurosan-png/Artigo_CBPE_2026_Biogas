# Alterações recomendadas no manuscrito v02

1. Substituir “despacho horário formulado como MILP com horizonte anual” por
   “simulação anual composta por MILPs sequenciais de 24 horas com continuidade
   dos estados”, na descrição do Pareto.
2. Chamar 468,25 kW de “menor pico observado para PV+BSV nas fronteiras
   avaliadas” enquanto o modelo anual integrado não comprovar o limiar.
3. Não usar falhas do processo `run_single_case` como casos inviáveis.
4. Explicitar que 351,25 kW é obtido com PV+BSV+H2 e BSV limitada a 1.500 kWh,
   portanto não contradiz o mínimo PV+BSV de 468,25 kW.
5. Delimitar a afirmação de LCOE de 0,15-0,18 USD/kWh ao cenário e critério de
   seleção correspondentes; os Pareto arquivados abrangem faixa maior.
6. Atualizar números, legendas e conclusões após a execução completa registrada
   no manifesto da release.
7. Não usar a execução anual preliminar de 120 segundos como prova: ela terminou
   por limite de tempo com gaps de 20% (PV+BSV) e 100% (PV+BSV+H2).
