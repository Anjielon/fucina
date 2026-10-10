"""Importers that pre-load the measurement database from existing lab records.

``labels`` maps free-text experiment labels to registered techniques (via each technique's
``measurement_aliases``). ``cycles`` and ``thor_notes`` parse the Thor-T lab notebook; their regular expressions match
that notebook's (Italian) wording on purpose, since they read data, not code. File locations come from the
``import_sources`` section of the configuration."""
