#!/bin/bash

# ═══════════════════════════════════════════════════════════
#  Mosaic Setup Script
#  Checks dependencies, creates venv, installs packages,
#  and validates environment configuration.
# ═══════════════════════════════════════════════════════════

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASS="${GREEN}✓${NC}"
FAIL="${RED}✗${NC}"
WARN="${YELLOW}⚠${NC}"

errors=0
warnings=0

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Mosaic Setup${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────
# 1. Check system dependencies
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[1/6] Checking system dependencies...${NC}"
echo ""

# Python
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "  ${PASS} Python ${PY_VERSION}"
else
    echo -e "  ${FAIL} Python 3 not found"
    echo "       Install: https://python.org/downloads"
    errors=$((errors + 1))
fi

# Node.js
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "  ${PASS} Node.js ${NODE_VERSION}"
else
    echo -e "  ${FAIL} Node.js not found"
    echo "       Install: https://nodejs.org"
    errors=$((errors + 1))
fi

# npm
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "  ${PASS} npm ${NPM_VERSION}"
else
    echo -e "  ${FAIL} npm not found (comes with Node.js)"
    errors=$((errors + 1))
fi

# Ollama
if command -v ollama &> /dev/null; then
    echo -e "  ${PASS} Ollama installed"
    # Check if running
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "  ${PASS} Ollama server running"
        # Check for models
        MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('models',[])))" 2>/dev/null || echo "0")
        if [ "$MODELS" -gt "0" ]; then
            echo -e "  ${PASS} ${MODELS} model(s) available"
        else
            echo -e "  ${WARN} No models pulled. Run: ollama pull mistral"
            warnings=$((warnings + 1))
        fi
    else
        echo -e "  ${WARN} Ollama not running. Start with: ollama serve"
        warnings=$((warnings + 1))
    fi
else
    echo -e "  ${WARN} Ollama not installed (needed for local LLM)"
    echo "       Install: https://ollama.com"
    warnings=$((warnings + 1))
fi

# Git
if command -v git &> /dev/null; then
    echo -e "  ${PASS} Git $(git --version | awk '{print $3}')"
else
    echo -e "  ${WARN} Git not found (optional, for version control)"
    warnings=$((warnings + 1))
fi

echo ""

# ─────────────────────────────────────────────────────────
# 2. Setup Python virtual environment
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[2/6] Setting up Python virtual environment...${NC}"
echo ""

if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    echo -e "  ${PASS} Virtual environment exists"
else
    echo "  Creating .venv..."
    python3 -m venv .venv
    echo -e "  ${PASS} Virtual environment created"
fi

# Activate
source .venv/bin/activate
echo -e "  ${PASS} Activated (.venv/bin/python)"

# Install requirements
echo "  Installing Python dependencies..."
pip install -r requirements.txt --quiet 2>&1 | tail -1
echo -e "  ${PASS} Python dependencies installed"

echo ""

# ─────────────────────────────────────────────────────────
# 3. Setup Frontend
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[3/6] Setting up Frontend...${NC}"
echo ""

cd Frontend
if [ -d "node_modules" ]; then
    echo -e "  ${PASS} node_modules exists"
else
    echo "  Installing npm packages..."
    npm install --silent 2>&1 | tail -1
    echo -e "  ${PASS} npm packages installed"
fi
cd ..

echo ""

# ─────────────────────────────────────────────────────────
# 4. Check Backend .env
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[4/6] Checking Backend environment...${NC}"
echo ""

if [ -f "Backend/.env" ]; then
    echo -e "  ${PASS} Backend/.env exists"
    
    # Check required vars
    check_env_var() {
        local file=$1
        local var=$2
        local required=$3
        if grep -q "^${var}=" "$file" && [ "$(grep "^${var}=" "$file" | cut -d= -f2-)" != "" ]; then
            echo -e "  ${PASS} ${var} is set"
        elif [ "$required" = "required" ]; then
            echo -e "  ${FAIL} ${var} is NOT set (required)"
            errors=$((errors + 1))
        else
            echo -e "  ${WARN} ${var} is not set (optional)"
            warnings=$((warnings + 1))
        fi
    }

    check_env_var "Backend/.env" "TAVILY_API_KEY" "required"
    check_env_var "Backend/.env" "ADMIN_PASSWORD" "required"
    check_env_var "Backend/.env" "USER_PASSWORD" "optional"
    check_env_var "Backend/.env" "JWT_SECRET" "optional"
else
    echo -e "  ${FAIL} Backend/.env not found"
    echo ""
    echo "       Create it from the example:"
    echo "         cp Backend/.env.example Backend/.env"
    echo "       Then fill in your API keys and passwords."
    errors=$((errors + 1))
fi

echo ""

# ─────────────────────────────────────────────────────────
# 5. Check Frontend .env
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[5/6] Checking Frontend environment...${NC}"
echo ""

if [ -f "Frontend/.env" ]; then
    echo -e "  ${PASS} Frontend/.env exists"
    check_env_var "Frontend/.env" "AUTH_SECRET" "required"
    check_env_var "Frontend/.env" "NEXT_PUBLIC_BACKEND_URL" "required"
    check_env_var "Frontend/.env" "BACKEND_URL" "required"
else
    echo -e "  ${FAIL} Frontend/.env not found"
    echo ""
    echo "       Create it from the example:"
    echo "         cp Frontend/.env.example Frontend/.env"
    echo "       Then set AUTH_SECRET (generate with: openssl rand -base64 32)"
    errors=$((errors + 1))
fi

echo ""

# ─────────────────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────────────────

echo -e "${BLUE}[6/6] Summary${NC}"
echo ""

if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo -e "  ${GREEN}Everything looks good!${NC}"
elif [ $errors -eq 0 ]; then
    echo -e "  ${YELLOW}Setup complete with ${warnings} warning(s).${NC}"
else
    echo -e "  ${RED}${errors} error(s) and ${warnings} warning(s) found.${NC}"
    echo "  Fix the errors above before running."
fi

echo ""
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}"
echo ""

if [ $errors -eq 0 ]; then
    echo "  To run Mosaic:"
    echo ""
    echo "    Terminal 1:  ollama serve"
    echo "    Terminal 2:  cd Backend && source ../.venv/bin/activate && uvicorn cifastapi_mosaic:app --port 8080"
    echo "    Terminal 3:  cd Frontend && npm run dev"
    echo ""
    echo "  Then open: http://localhost:3000"
fi

echo ""
