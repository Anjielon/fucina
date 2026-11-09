#!/bin/bash
# Riga per il ping: stato VERO delle unita' E2E attive o fallite di recente (attiva/fallita, riavvii, errore nel journal).
# Nato il 26/9 18:1x: il crash delle 17:15 stava solo nel journal (traceback), non nel registro.log -> visto 50 min dopo.
export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
for u in $(systemctl --user list-units 'e2e-*' --all --plain --no-legend 2>/dev/null | awk '{print $1}' | sed 's/\.service$//'); do
  s=$(systemctl --user show "$u" -p ActiveState -p NRestarts -p Result --value 2>/dev/null | tr '\n' ' ')
  err=$(journalctl --user -u "$u" --since "-15min" --no-pager 2>/dev/null | grep -E "Error|Traceback|FALLITO|Failed" | tail -1 | cut -c1-160)
  case "$s" in active*|activating*) ;; *) [ -z "$err" ] && continue ;; esac
  echo "$u: $s${err:+ ⛔ $err}"
done
echo "disco @@ROOT:lavoro@@ $(df -h --output=avail @@ROOT:lavoro@@ | tail -1 | tr -d ' ') liberi"
