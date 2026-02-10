#!/bin/bash
set -e

# Create test database snapshot from production database
PROD_DB="/penny/data/penny.db"
TEST_DB="/penny/data/penny-test.db"

if [ -f "$PROD_DB" ]; then
    echo "Creating test database snapshot: $TEST_DB"
    cp "$PROD_DB" "$TEST_DB"
else
    echo "Production database does not exist yet, skipping test snapshot"
fi

# Execute the main command
exec "$@"
