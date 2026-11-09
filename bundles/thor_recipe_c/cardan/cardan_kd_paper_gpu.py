#!/usr/bin/env python3
"""Trainer paper originale con runtime GPU opt-in e diagnostica caricamenti."""
import sys
from cardan_allena_runtime_gpu import esegui

if __name__ == "__main__":
    sys.exit(esegui("cardan_kd_paper"))
