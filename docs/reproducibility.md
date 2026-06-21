# Reprodutibilidade

## Ambiente

Python 3.11 a 3.13 é suportado. A release de referência usa as versões exatas
de `requirements.lock`; o `Dockerfile` fornece o ambiente Linux equivalente.

## Perfis

`python -m cbpe reproduce --profile smoke` valida a instalação em poucos
minutos. Ele usa 48 horas para a análise integrada de viabilidade e recria os
artefatos canônicos a partir dos dados-fonte rastreados.

`python -m cbpe reproduce --profile paper` executa quatro buscas NSGA-II e duas
análises anuais integradas. Os resultados completos, logs e despachos devem ser
anexados à GitHub Release; apenas tabelas, figuras e manifesto ficam no Git.

## Interpretação do horizonte

A busca de Pareto avalia o ano por MILPs consecutivos de 24 horas, propagando
SOC da bateria, estoque de H2 e pico acumulado. Portanto, não equivale a um
único MILP anual com antecipação perfeita. A otimização específica de
viabilidade usa um modelo anual integrado e impõe estoque final não inferior ao
inicial.

## Verificação

Cada manifesto registra commit, seed 42, versões, plataforma, hash dos dados,
estado do solver, objetivo, melhor limite, gap, duração e hashes dos artefatos.

```bash
python -m cbpe verify results/paper/manifest.json
```

