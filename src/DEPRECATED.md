# [HISTÓRICO] Substituído por `voice/`

Este pacote foi a **primeira versão** do modelo de voz: um classificador
só de direção (`up`, `down`, `left`, `right`, `stop`, `silence`,
`unknown` — 7 classes).

Ele não é mais usado pelo jogo. Foi substituído por `voice/`, um modelo
único que também reconhece números, depois que testes ao vivo mostraram
confusão real entre este modelo e `digits/` (o segundo modelo, separado,
de números) — cada um rodava num microfone próprio e nunca tinha
aprendido de verdade o vocabulário do outro.

Mantido no repositório de propósito: o processo de diagnosticar esse
problema e corrigi-lo estruturalmente é conteúdo de metodologia do TCC.
Veja o README na raiz do projeto para o histórico completo.