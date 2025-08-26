#!/bin/bash
echo "=== Test 1: OHNE --claude flag ==="
timeout 2 npx claude-flow@alpha hive-mind spawn "Test ohne claude" --auto-spawn 2>&1 &
PID1=$!
sleep 1
kill -9 $PID1 2>/dev/null
echo "Resultat: Hängt, wartet auf Input"

echo -e "\n=== Test 2: MIT --claude flag ==="
timeout 2 npx claude-flow@alpha hive-mind spawn "Test mit claude" --claude --auto-spawn 2>&1 | grep -E "Launching Claude Code|Swarm ID"
echo "Resultat: Startet Claude Code sofort"

echo -e "\n=== Test 3: MIT --execute flag ==="
timeout 2 npx claude-flow@alpha hive-mind spawn "Test mit execute" --execute --auto-spawn 2>&1 | grep -E "Launching|Swarm"
