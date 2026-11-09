#!/usr/bin/env python3
"""Controllo/trainer KL originale con runtime mmap opt-in."""
import sys
from cardan_allena_runtime_gpu import esegui

if __name__ == "__main__":
    sys.exit(esegui("cardan_kd"))
