# TCC: Classificação de Comandos de Voz em Tempo Real com PyTorch

Proof of concept de reconhecimento de comandos de voz em tempo real
(PyTorch), integrado a dois jogos: um **Pong** controlado por voz e um
**jogo de tabuleiro** (estilo Mario Party) com labirinto gerado
proceduralmente e um sistema de cartas jogado por voz.

## Sobre o autor e o contexto do projeto

Sou sound designer de jogos, e esta foi minha primeira experiência
aprofundada com desenvolvimento de jogos e machine learning. O projeto
nasceu de uma ideia maior, ligada à minha área: um jogo estilo "Marco
Polo" jogado inteiramente por áudio (sem visão), com ambientação sonora
3D e reconhecimento de voz, pensado com acessibilidade em mente. Dado o
escopo de um PoC, o projeto foi reduzido ao componente técnico central
dessa ideia — o classificador de comandos de voz em tempo real — testado
em jogos mais simples (Pong e um jogo de tabuleiro) em vez do jogo de
áudio 3D completo. Veja "Trabalhos futuros" no final deste documento.

## Objetivo

Investigar a viabilidade de usar um classificador de comandos de voz
(CNN sobre espectrograma log-mel, treinado do zero em PyTorch) para
controlar mecânicas de jogo em tempo real, com precisão e latência
suficientes para uso prático — sem depender de dados de voz coletados
pelos autores, apenas de dataset público, testando a capacidade do
sistema de generalizar para qualquer falante.

**Objetivos específicos:**
1. Treinar e avaliar um classificador de voz com dataset público
   (Google Speech Commands), validando a generalização por meio de um
   split de teste que nunca reutiliza a voz de um falante já visto no
   treino.
2. Integrar o reconhecimento em tempo real a duas aplicações interativas
   diferentes (um jogo de reflexo simples e um jogo de tabuleiro com
   decisão estratégica), verificando se a arquitetura se sustenta em
   cenários de uso real, não só em teste isolado.
3. Diagnosticar e corrigir falhas reais encontradas em teste ao vivo
   (não previstas pela métrica de teste isolada) através de iteração —
   do diagnóstico à correção estrutural, documentando o processo.

**Por que isso importa:** voz como forma de interação em jogos vai além
de comodidade — é um mecanismo de acessibilidade (jogos jogáveis sem uso
das mãos ou da visão) e uma linha de design de interação pouco
explorada fora de assistentes virtuais genéricos. Este projeto não
implementa uma aplicação de acessibilidade completa, mas valida o
componente técnico central que qualquer aplicação desse tipo
precisaria: um classificador de comandos de voz confiável, rodando em
tempo real, com pipeline de dados e treino reprodutível do início ao
fim.

## Estado atual do projeto

O projeto passou por uma iteração real de desenvolvimento, documentada
abaixo porque isso é conteúdo legítimo de metodologia pro TCC (não é
só "o resultado final", é o processo).

| Pasta | Status | O que é |
|---|---|---|
| **`voice/`** | ✅ **Em uso** | Modelo único e unificado (13 classes: `up/down/left/right/stop` + `one`-`six` + `silence` + `unknown`). É o que os dois jogos usam hoje. |
| `src/` | 🗄️ Histórico | Primeira versão: modelo só de direção (5 classes). Substituído. |
| `digits/` | 🗄️ Histórico | Segunda versão: modelo separado só de números (8 classes), rodando em paralelo ao `src/`. Substituído. |

**Por que `src/` e `digits/` existem mas não são usados**: a abordagem
inicial usava dois classificadores especialistas rodando ao mesmo tempo
(dois microfones simultâneos). Em teste ao vivo, isso causou confusão
real e reproduzível (ex: a palavra `"left"` sendo classificada como
`"one"`, `"right"` como `"five"`) — cada modelo nunca tinha aprendido de
verdade o vocabulário do outro, só uma amostra fraca dele como classe
negativa. Tentativas de correção incremental (mais dados negativos, peso
de classe, limiares de confiança) reduziram mas não eliminaram o
problema. A correção estrutural foi unificar tudo num único modelo de 13
classes (`voice/`), que aprende cada palavra como classe própria e
balanceada — isso eliminou a confusão por completo em teste. Essa
progressão (diagnóstico → tentativa incremental → correção estrutural)
está documentada nos commits do repositório.

## Os dois jogos

- **`pong_voice.py`** — Pong clássico, raquete A controlada por voz
  (`up`/`down`/`left`/`right`/`stop`), raquete B por teclado.
- **`boardgame/`** — jogo de tabuleiro: leve seu peão do ponto A ao ponto
  B, desviando de paredes (bloqueiam e fazem "quicar") e buracos
  (mandam de volta ao início). Direção falada livremente; número de casas
  escolhido de uma mão de 3 cartas (também falada), sempre garantida
  para não te forçar a cair num buraco. Labirinto gerado
  proceduralmente, com garantia matemática (BFS) de que sempre existe
  caminho A→B.

## Pipeline do modelo de voz (`voice/`)

```bash
# 0. Ambiente
python -m venv venv
venv\Scripts\activate            # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# 1. Importar o vocabulário completo do dataset público
#    (baixe e extraia http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz antes)
python -m voice.prepare_dataset --source_dir /caminho/para/speech_commands_v0.02_extracted --unknown_per_word 220

# 2. Extrair ruído real de fundo para a classe "silence"
python -m voice.extract_background_noise --source_dir /caminho/para/speech_commands_v0.02_extracted --num 400

# 3. Treinar
python -m voice.train
#    -> voice/checkpoints/best_model.pt
#    -> voice/results/train_log.csv, training_curves.png

# 4. Avaliar no conjunto de teste (falantes nunca vistos no treino)
python -m voice.evaluate
#    -> voice/results/confusion_matrix.png
#    -> voice/results/classification_report.txt
#    -> voice/results/inference_latency.csv

# 5. Testar o reconhecimento em tempo real isoladamente (sem jogo)
python -m voice.bridge

# 6. Jogar
python pong_voice.py
python -m boardgame.game
```

## Estrutura do projeto

```
puc-rio-ia-generativa-llm/
├── voice/                        # modelo unificado (EM USO)
│   ├── config.py
│   ├── prepare_dataset.py        # importa up/down/left/right/stop/one-six + unknown
│   ├── extract_background_noise.py  # silence a partir de ruído real
│   ├── features.py               # waveform -> log-mel spectrogram
│   ├── dataset.py                # Dataset + split por falante
│   ├── model.py                  # CNN (13 classes)
│   ├── train.py                  # treino com class-weighted loss
│   ├── evaluate.py               # matriz de confusão, latência
│   └── bridge.py                 # listener em tempo real (thread + fila)
├── boardgame/                    # jogo de tabuleiro
│   ├── config.py
│   ├── maze.py                   # geração de labirinto + física de movimento/quique
│   ├── deck.py                   # sistema de cartas de número
│   └── game.py                   # loop principal (Turtle)
├── pong_voice.py                 # Pong controlado por voz
├── pong_original.py              # arquivo original do usuário, intocado
├── src/                          # [histórico] modelo só de direção
├── digits/                       # [histórico] modelo só de número
├── data/, results/, checkpoints/ # artefatos do src/ (histórico)
└── requirements.txt
```

## Resultados

Tudo em `voice/results/` vem de rodar os scripts de verdade — nenhum
número foi inventado.

- `voice/results/dataset_manifest.csv` → composição do dataset.
- `voice/results/training_curves.png` → curvas de loss/acurácia.
- `voice/results/confusion_matrix.png` + `classification_report.txt` → acurácia por classe.
- `voice/results/inference_latency.csv` → viabilidade de tempo real.
- O histórico de commits do repositório documenta o processo iterativo de diagnóstico e correção (dois modelos → confusão observada → correção estrutural).

**Limitações a declarar honestamente:**
1. Dataset é público (Google Speech Commands), não gravado pelos autores — é um PoC de viabilidade, generalização a qualquer falante, não uma validação de uso pessoal.
2. `silence` usa ruído de fundo real do próprio dataset (não do ambiente de uso final) — ainda pode não cobrir todo tipo de ruído do mundo real.
3. `left`/`right` no Pong são um placeholder de movimentação (ver comentário em `pong_voice.py`).
4. Vocabulário de comando é em inglês (herdado do dataset), não português.

## Trabalhos futuros

O objetivo original — um jogo de "Marco Polo" jogado inteiramente por
áudio, com ambientação sonora 3D (via FMOD) dentro da Unreal Engine, e
justificativa de acessibilidade — não foi implementado neste PoC por
questão de escopo e prazo. O classificador de voz validado aqui é o
componente técnico que sustentaria essa aplicação: o caminho natural de
continuação seria exportar o modelo treinado para ONNX e integrá-lo à
Unreal via seu plugin de rede neural (NNE), mantendo o pipeline de
treino em PyTorch como está.

## Dataset e licença

[Google Speech Commands Dataset v0.02](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)
(Warden, P. *Speech Commands: A public dataset for single-word speech
recognition*, 2017), licença Creative Commons BY 4.0.