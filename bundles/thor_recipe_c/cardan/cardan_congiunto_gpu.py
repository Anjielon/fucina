#!/usr/bin/env python3
"""Trainer congiunto originale; nessuno shard indipendente della loss."""
import sys
from cardan_allena_runtime_gpu import esegui

if __name__ == "__main__":
    sys.exit(esegui("cardan_congiunto"))
