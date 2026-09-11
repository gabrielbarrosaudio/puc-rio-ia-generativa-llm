# [HISTÓRICO] Substituído por `voice/`

Este pacote foi a **segunda versão** do modelo de voz: um classificador
só de números (`one`-`six`, `silence`, `unknown` — 8 classes), treinado e
rodando separadamente do modelo de direção em `src/`.

Ele não é mais usado pelo jogo. Rodar dois modelos especialistas em
paralelo (dois microfones simultâneos) causou confusão real em teste ao
vivo -- por exemplo, a palavra `"left"` sendo reconhecida como `"one"`,
porque este modelo nunca tinha aprendido de verdade a rejeitar palavras
de direção como negativas (só uma amostra fraca delas). Foi substituído
por `voice/`, um único modelo com as 13 classes (direção + número)
aprendidas juntas e balanceadas.

Mantido no repositório de propósito: o processo de diagnosticar esse
problema e corrigi-lo estruturalmente é conteúdo de metodologia do TCC.
Veja o README na raiz do projeto para o histórico completo.