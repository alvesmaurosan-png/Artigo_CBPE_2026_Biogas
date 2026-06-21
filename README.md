# CBPE 2026: planejamento energético de garagens de ônibus elétricos

Repositório auditável do artigo sobre microrredes PV-BSV-H2 sob restrições de
potência. O código separa a busca multiobjetivo da evidência formal de
viabilidade e registra dados, configurações, ambiente e artefatos em um
manifesto verificável.

## Início rápido

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.lock
pip install -e .
python -m cbpe reproduce --profile smoke
python -m cbpe verify results/paper/manifest.json
```

O perfil `smoke` valida dados, executa dois casos reduzidos de viabilidade e
regenera as quatro figuras e a Tabela 2. O perfil completo executa também os
quatro cenários NSGA-II e a análise anual:

```bash
python -m cbpe reproduce --profile paper
```

Essa execução é computacionalmente intensiva. Use o perfil completo para uma
release científica; a integração contínua usa o perfil rápido.

## Método e proveniência

- O Pareto usa NSGA-II e uma simulação anual composta por MILPs sequenciais de
  24 horas, com continuidade do estado dos armazenamentos.
- O limiar de potência é investigado separadamente por um MILP anual integrado
  de dimensionamento e despacho, que registra melhor solução, limite inferior
  e gap.
- `468,25 kW` é tratado como o mínimo observado no Pareto PV+BSV até a
  otimização anual confirmar a interpretação como limiar estrutural.
- Dados, cenários e correspondência com o manuscrito estão documentados em
  `data/README.md`, `docs/reproducibility.md` e `AUDIT_REPORT.md`.

## English

This repository contains the auditable computational package for the CBPE
paper on power-constrained PV-BSV-H2 electric bus depots. Install the locked
environment and run `python -m cbpe reproduce --profile smoke` for a quick
check or `--profile paper` for the computationally intensive full workflow.
Every run writes a hash-verifiable manifest under `results/paper/`.

Software is MIT licensed. Data and scientific documentation are CC BY 4.0.

