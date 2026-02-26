#!/usr/bin/env bash
# Habla database backup script.
# Creates a gzipped SQLite dump with timestamp.
#
# Usage:
#   ./scripts/backup_db.sh                  # uses default paths
#   ./scripts/backup_db.sh /path/to/habla.db /path/to/backups/
#
# Restore:
#   gunzip -k backup_file.sql.gz
#   sqlite3 habla_restored.db < backup_file.sql
#
# Cron example (daily at 2 AM):
#   0 2 * * * cd /path/to/habla && ./scripts/backup_db.sh

set -euo pipefail

DB_PATH="${1:-data/habla.db}"
BACKUP_DIR="${2:-data/backups}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="${BACKUP_DIR}/habla_${TIMESTAMP}.sql.gz"
MAX_BACKUPS=30

if [ ! -f "$DB_PATH" ]; then
    echo "ERROR: Database not found at $DB_PATH"
    exit 1
fi

mkdir -p "$BACKUP_DIR"

# Use .dump for a consistent backup (safe even while server is running with WAL mode)
sqlite3 "$DB_PATH" .dump | gzip > "$BACKUP_FILE"

SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
echo "Backup created: $BACKUP_FILE ($SIZE)"

# Rotate old backups — keep only the newest MAX_BACKUPS
COUNT=$(ls -1 "${BACKUP_DIR}"/habla_*.sql.gz 2>/dev/null | wc -l)
if [ "$COUNT" -gt "$MAX_BACKUPS" ]; then
    REMOVE=$((COUNT - MAX_BACKUPS))
    ls -1t "${BACKUP_DIR}"/habla_*.sql.gz | tail -n "$REMOVE" | xargs rm -f
    echo "Rotated $REMOVE old backup(s), keeping $MAX_BACKUPS"
fi
