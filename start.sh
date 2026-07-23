#!/bin/bash

# ═══════════════════════════════════════════════════════════
#  Mosaic Start Script
#  Starts all services in background and shows the URL.
#  Usage: ./start.sh
#  Stop:  ./start.sh stop
# ═══════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_DIR="$ROOT_DIR/.pids"
LOG_DIR="$ROOT_DIR/Backend/logs"

mkdir -p "$PID_DIR" "$LOG_DIR"

# ─────────────────────────────────────────────────────────
# Stop command
# ─────────────────────────────────────────────────────────

stop_all() {
    echo ""
    echo -e "${BLUE}Stopping Mosaic services...${NC}"
    
    # Kill by PID file if available
    if [ -f "$PID_DIR/backend.pid" ]; then
        kill $(cat "$PID_DIR/backend.pid") 2>/dev/null
        rm -f "$PID_DIR/backend.pid"
    fi
    if [ -f "$PID_DIR/frontend.pid" ]; then
        kill $(cat "$PID_DIR/frontend.pid") 2>/dev/null
        rm -f "$PID_DIR/frontend.pid"
    fi
    if [ -f "$PID_DIR/ollama.pid" ]; then
        kill $(cat "$PID_DIR/ollama.pid") 2>/dev/null
        rm -f "$PID_DIR/ollama.pid"
    fi

    # Also kill by port (catches manually started processes)
    lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null
    lsof -ti:3000 2>/dev/null | xargs kill -9 2>/dev/null

    echo -e "  ${GREEN}✓${NC} Backend (port 8080) stopped"
    echo -e "  ${GREEN}✓${NC} Frontend (port 3000) stopped"
    echo ""
    exit 0
}

if [ "$1" = "stop" ]; then
    stop_all
fi

# ─────────────────────────────────────────────────────────
# Start
# ─────────────────────────────────────────────────────────

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Starting Mosaic${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo ""

# 1. Ollama
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Ollama already running"
else
    echo -n "  Starting Ollama... "
    ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    echo $! > "$PID_DIR/ollama.pid"
    sleep 2
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
    else
        echo -e "${RED}✗ Failed to start${NC}"
        echo "    Check: $LOG_DIR/ollama.log"
    fi
fi

# 2. Backend
echo -n "  Starting Backend (port 8080)... "
cd "$ROOT_DIR/Backend"
source "$ROOT_DIR/.venv/bin/activate"
uvicorn cifastapi_mosaic:app --port 8080 > "$LOG_DIR/backend_stdout.log" 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PID_DIR/backend.pid"
cd "$ROOT_DIR"

# Wait for backend to be ready
for i in {1..15}; do
    if curl -s http://localhost:8080/health &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${RED}✗ Timed out${NC}"
        echo "    Check: $LOG_DIR/backend_stdout.log"
    fi
    sleep 1
done

# 3. Frontend
echo -n "  Starting Frontend (port 3000)... "
cd "$ROOT_DIR/Frontend"
npm run dev > "$LOG_DIR/frontend_stdout.log" 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > "$PID_DIR/frontend.pid"
cd "$ROOT_DIR"

# Wait for frontend
for i in {1..15}; do
    if curl -s http://localhost:3000 &> /dev/null; then
        echo -e "${GREEN}✓${NC}"
        break
    fi
    if [ $i -eq 15 ]; then
        echo -e "${YELLOW}⚠ Still starting (may take a moment)${NC}"
    fi
    sleep 1
done

# ─────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────

echo ""
echo -e "${BLUE}─────────────────────────────────────────────────────────${NC}"
echo ""
echo -e "  ${GREEN}Mosaic is running!${NC}"
echo ""
echo -e "  🌐  ${GREEN}http://localhost:3000${NC}"
echo ""
echo -e "  Logs:"
echo "    Backend:   $LOG_DIR/backend_stdout.log"
echo "    Frontend:  $LOG_DIR/frontend_stdout.log"
echo ""
echo -e "  To stop:  ${YELLOW}./start.sh stop${NC}"
echo ""
