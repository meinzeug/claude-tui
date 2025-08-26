#!/bin/bash

# AI SOFTWARE EMPIRE LAUNCHER
# The Ultimate CEO Command Center

set -e

# Colors
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m'

clear
echo -e "${CYAN}"
cat << "EOF"
   ______   ______   ____       ______                                          __
  / ____/  / ____/  / __ \     / ____/___   ____ ___   ____ ___   ____ _ ____  ____/ /
 / /      / __/    / / / /    / /    / _ \ / __ `__ \ / __ `__ \ / __ `// __ \/ __  / 
/ /___   / /___   / /_/ /    / /___ /  __// / / / / // / / / / // /_/ // / / / /_/ /  
\____/  /_____/   \____/     \____/ \___//_/ /_/ /_//_/ /_/ /_/ \__,_//_/ /_/\__,_/   
                                                                                        
EOF
echo -e "${NC}"
echo -e "${BOLD}Welcome to Your AI Software Empire!${NC}"
echo ""

# Function to check if claude-flow is available
check_claude_flow() {
    if command -v claude-flow &> /dev/null || npx claude-flow@alpha --version &> /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Main menu
show_menu() {
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -e "${BOLD}    CEO COMMAND CENTER${NC}"
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo ""
    echo "1. 🚀 Start New Project"
    echo "2. 👥 Executive Meeting"
    echo "3. 📊 Company Dashboard"
    echo "4. 💼 Department Status"
    echo "5. 🏭 Production Pipeline"
    echo "6. 💰 Financial Overview"
    echo "7. 📈 Market Analysis"
    echo "8. 🔧 DevOps Center"
    echo "9. 🛡️ Security Audit"
    echo "10. 📝 Generate Report"
    echo ""
    echo "0. Exit"
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════${NC}"
    echo -n "CEO Decision: "
}

# Start new project with full company involvement
start_project() {
    echo -e "\n${BOLD}${GREEN}INITIATING NEW PROJECT${NC}"
    echo ""
    read -p "Project Name: " PROJECT_NAME
    read -p "Project Description: " PROJECT_DESC
    read -p "Budget (€): " BUDGET
    read -p "Deadline (days): " DEADLINE
    
    echo ""
    echo -e "${YELLOW}Assembling AI Team...${NC}"
    
    if check_claude_flow; then
        # Use Claude-Flow for orchestration
        echo -e "${GREEN}Using Claude-Flow Hive Mind${NC}"
        
        # Fix database before running
        mkdir -p .hive-mind .swarm
        
        # Create project with hive mind
        npx -y claude-flow@alpha hive-mind spawn "Create a new software project: $PROJECT_NAME. Description: $PROJECT_DESC. Budget: $BUDGET. Deadline: $DEADLINE days. Departments involved: Engineering, QA, DevOps, Security, UX. Create full implementation plan." --claude --auto-spawn || {
            echo "Falling back to Python implementation..."
            python3 company/run_project.py "$PROJECT_NAME" "$PROJECT_DESC" "$BUDGET" "$DEADLINE"
        }
    else
        # Fallback to Python
        echo -e "${YELLOW}Using Python AI Company (Fallback)${NC}"
        python3 company/run_project.py "$PROJECT_NAME" "$PROJECT_DESC" "$BUDGET" "$DEADLINE"
    fi
}

# Executive meeting room
executive_meeting() {
    echo -e "\n${BOLD}${MAGENTA}EXECUTIVE MEETING ROOM${NC}"
    echo ""
    echo "Participants: CEO (You), CTO, CFO, COO, CMO, CHRO, CIO"
    echo ""
    read -p "Meeting Agenda: " AGENDA
    
    if check_claude_flow; then
        npx -y claude-flow@alpha swarm "Conduct executive meeting with agenda: $AGENDA. Participants: CTO, CFO, COO, CMO, CHRO, CIO. Each executive should provide input from their department perspective. Generate actionable decisions."
    else
        python3 company/executive_meeting.py "$AGENDA"
    fi
}

# Main loop
while true; do
    show_menu
    read choice
    
    case $choice in
        1) start_project ;;
        2) executive_meeting ;;
        3) python3 company/dashboard.py ;;
        4) python3 company/departments.py ;;
        5) python3 company/pipeline.py ;;
        6) python3 company/finance.py ;;
        7) python3 company/market.py ;;
        8) python3 company/devops.py ;;
        9) python3 company/security.py ;;
        10) python3 company/reports.py ;;
        0) echo "CEO signing off. Company continues operating..."; exit 0 ;;
        *) echo "Invalid choice" ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
    clear
done
