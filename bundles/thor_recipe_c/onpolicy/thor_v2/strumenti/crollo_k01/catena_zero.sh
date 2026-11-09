#!/bin/bash
# zero-shot MMLU: maestro subito, K01 dopo il suo dump principale, base dopo il suo
cd @@ROOT:depot@@/diagnosi_k01 || exit 2
PY=@@ROOT:venv@@/bin/python3; CK=@@ROOT:forge@@/thor_v2/crollo_k01.py
$PY $CK zero maestro > zero_maestro.log 2>&1; echo "maestro rc=$?"
until [ -f llama_k01/manifesto.json ]; do sleep 20; done
$PY $CK zero k01 > zero_k01.log 2>&1; echo "k01 rc=$?"
until [ -f llama_base/manifesto.json ]; do sleep 20; done
$PY $CK zero base > zero_base.log 2>&1; echo "base rc=$?"
