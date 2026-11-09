#!/bin/bash
# 27/9 15:5x — UN solo lavoro alla volta su Aragorn (RTX 4060 8 GB, ~9 GB di disco libero): due MMLU insieme non ci stanno
# e la pulizia dei GGUF di uno cancellerebbe il file dell'altro. Uso: aragorn_lucchetto.sh <comando> [argomenti...]
exec flock @@ROOT:forge@@/stato/aragorn.lock "$@"
