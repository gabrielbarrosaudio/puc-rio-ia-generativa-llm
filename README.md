# TCC: Classificação de Comandos de Voz em Tempo Real com PyTorch, Integrada a um Jogo Pong

Proof of concept: um CNN treinado em PyTorch classifica comandos de voz
(`up`, `down`, `left`, `right`, `stop`) em tempo real e controla a raquete A
de um jogo Pong feito com Turtle.

## Escopo (decidido nesta conversa)

- **Dataset**: público, [mini_speech_commands](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html)
  (Warden, 2018) -- 1000 clipes cada de `up/down/left/right/stop`, mais
  `go/no/yes` reaproveitados como classe `unknown`. Nenhuma voz sua é
  necessária -- é um PoC deliberadamente treinado só com dados públicos,
  para generalizar a qualquer falante.
- **Silêncio**: sintético (ruído gaussiano de baixa amplitude), já que o
  dataset público não inclui essa classe. **Documente isso como limitação
  conhecida no TCC** -- silêncio sintético não é idêntico a ruído real de
  ambiente.
- **Split treino/val/teste**: por falante (nenhuma pessoa aparece em mais
  de um conjunto), pra medir generalização de verdade a vozes não vistas.

## Pipeline, passo a passo

```bash
# 0. Ambiente
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 1. Importar o dataset público (extraia o zip em algum lugar antes)
python -m src.prepare_public_dataset --source_dir /caminho/para/mini_speech_commands --unknown_per_word 400

# 2. Gerar a classe "silence" sintética
python -m src.generate_silence --num 300

# 3. Treinar
python -m src.train
#    -> checkpoints/best_model.pt
#    -> results/train_log.csv, results/training_curves.png

# 4. Avaliar no conjunto de teste (falantes nunca vistos no treino)
python -m src.evaluate
#    -> results/confusion_matrix.png
#    -> results/classification_report.txt
#    -> results/inference_latency.csv

# 5. Testar o reconhecimento em tempo real isoladamente (sem o jogo)
python -m src.bridge
#    fala um comando, deve aparecer "-> recognized: <label>" no terminal

# 6. Jogar com controle por voz
python pong_voice.py
```

## Estrutura do projeto

```
tcc_voice_pong/
├── src/
│   ├── config.py                 # todas as constantes centralizadas
│   ├── prepare_public_dataset.py # Stage 1b: importa o dataset público
│   ├── generate_silence.py       # Stage 1c: gera classe "silence" sintética
│   ├── record_data.py            # Stage 1 (opcional/futuro): gravação pessoal
│   ├── features.py               # Stage 2: waveform -> log-mel spectrogram
│   ├── dataset.py                # Stage 2: PyTorch Dataset + split por falante
│   ├── model.py                  # Stage 3: CNN
│   ├── train.py                  # Stage 4: treino, logging, checkpoints
│   ├── evaluate.py               # Stage 4b: matriz de confusão, latência
│   └── bridge.py                 # Stage 5: listener em tempo real (thread + fila)
├── pong_voice.py                 # Stage 6: jogo integrado
├── pong_original.py              # seu arquivo original, intocado
├── data/commands/<label>/*.wav   # populado pelos scripts acima
├── results/                      # métricas, gráficos, logs (gerados, não fabricados)
└── checkpoints/                  # pesos do modelo treinado
```

## Escrevendo os resultados no TCC

Tudo em `results/` vem de rodar os scripts de verdade -- nada foi
inventado. Sugestão de como usar cada artefato na monografia:

- `results/dataset_manifest.csv` → tabela de composição do dataset (seção de Metodologia).
- `results/training_curves.png` → curvas de loss/acurácia (mostra se houve overfitting).
- `results/confusion_matrix.png` + `classification_report.txt` → seção de Resultados (acurácia por classe, precision/recall/F1).
- `results/inference_latency.csv` → discussão de viabilidade de tempo real (latência de inferência pura, em CPU).
- Ao rodar `pong_voice.py`, anote observações qualitativas (falsos positivos durante ruído, delay percebido) -- isso é dado experimental real, mesmo sem uma métrica numérica formal.

**Limitações a declarar honestamente:**
1. Classe `silence` é sintética, não ruído real de ambiente.
2. O modelo nunca foi testado com a sua própria voz/microfone -- é um PoC de viabilidade, não uma validação de uso pessoal.
3. `left`/`right` no jogo são um placeholder de movimentação (ver comentário em `pong_voice.py`) até você redesenhar a raquete A.
