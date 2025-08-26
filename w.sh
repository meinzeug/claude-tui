npx claude-flow@alpha hive-mind spawn "Du bist die led queen. spawne dir ein mittelgroßes Team. 
Analysiere die vollständige Projektdokumentation und entwickle das Projekt weiter. Arbeite iterativ und systematisch basierend auf:

Alles aus PROJEKTDOKUMENTATION lesen und analysieren: ~/claude-tui/docs/*.md

AKTUELLE CODEBASE: Analysiere die aktuelle Codebase


ENTWICKLUNGSAUFGABEN:
1. Analysiere alle Markdown-Dateien in docs/ um den aktuellen Projektstand zu verstehen
2. Prüfe die bestehende Codebase in src/ (falls vorhanden)
3. Identifiziere die nächsten logischen Entwicklungsschritte basierend auf der Roadmap
4. Implementiere Features systematisch according to architecture.md
5. Erstelle/aktualisiere Tests basierend auf testing-strategy.md
6. Halte dich an die Spezifikationen in requirements.md und api-specification.md
7. Dokumentiere alle Änderungen und aktualisiere relevante .md Dateien

Arbeite systematisch und dokumentiere alle Schritte. Fokussiere auf qualitativ hochwertigen, produktionstauglichen Code.

Die Dokumentation für claude-flow findest du unter: https://github.com/ruvnet/claude-flow/tree/main/docs
Die Dokumentation für claued-code findest du unter: https://github.com/anthropics/claude-code

In der .gitlogin findest du folgende Variablen:
-für Github: GITHUB_TOKEN (PTA Acces Token), GITHUB_USER (Github User), GITHUB_REPO (Github Repository Name)

In der Daite .sp ist das Sudo Passwort, falls nötig

In der Datei .oa ist der Claude OAUTH Token gespeichert.

Spawne deine Agenten immer mit dem richtigen Agent type:
Available agents: general-purpose, statusline-setup, output-style-setup, refinement, pseudocode, specification, architecture, base-template-generator, crdt-synchronizer, raft-manager, 
     security-manager, gossip-coordinator, quorum-manager, byzantine-coordinator, performance-benchmarker, cicd-engineer, reviewer, tester, planner, coder, researcher, adaptive-coordinator, mesh-coordinator, 
     hierarchical-coordinator, system-architect, api-docs, backend-dev, ml-developer, tdd-london-swarm, production-validator, mobile-dev, pr-manager, repo-architect, multi-repo-swarm, sync-coordinator, github-modes, release-swarm, 
     workflow-automation, project-board-sync, swarm-pr, code-review-swarm, swarm-issue, issue-tracker, release-manager, sparc-coord, perf-analyzer, sparc-coder, task-orchestrator, migration-planner, smart-agent, memory-coordinator, 
     swarm-init, code-analyzer
" --claude --auto-spawn --continue --resume
