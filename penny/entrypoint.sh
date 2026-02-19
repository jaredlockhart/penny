#!/bin/bash
set -e

PROD_DB="/penny/data/penny.db"

if [ -f "$PROD_DB" ]; then
    # Create timestamped backup of production database
    BACKUP_DIR="/penny/data/backups"
    mkdir -p "$BACKUP_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/penny.db.$TIMESTAMP"
    echo "Creating database snapshot: $BACKUP_FILE"
    cp "$PROD_DB" "$BACKUP_FILE"

    # Keep only last 5 backups
    ls -t "$BACKUP_DIR"/penny.db.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null

    # Create test database snapshot from production database
    TEST_DB="/penny/data/penny-test.db"
    echo "Creating test database snapshot: $TEST_DB"
    cp "$PROD_DB" "$TEST_DB"
else
    echo "Production database does not exist yet, skipping snapshots"
fi

# Execute the main command
exec "$@"
