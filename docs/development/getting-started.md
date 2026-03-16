# Getting Started

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend)
- Git

## Setup

```bash
# Clone
git clone <repo-url>
cd mr_poker

# Install Python dependencies
pip install -e .[dev]

# Install frontend dependencies
cd apps/web_ui
npm install
cd ../..

# Copy environment config
cp .env.example .env
```

## Running

### API Server
```bash
python -m uvicorn apps.api.main:create_app --factory --reload
```

### Frontend Dev Server
```bash
cd apps/web_ui
npm run dev
```

### Run Tests
```bash
# All tests
bash infra/scripts/run_unit_tests.sh

# Specific test file
pytest tests/unit/test_engine_setup.py -v

# Property tests
pytest tests/property/ -v

# E2E tests
pytest tests/e2e/ -v
```

## Project Structure

```
mr_poker/
├── apps/
│   ├── api/          # FastAPI REST API
│   │   ├── main.py   # App factory
│   │   ├── container.py  # Dependency injection
│   │   ├── contracts.py  # Request models
│   │   ├── routers/  # API endpoints
│   │   └── middleware/   # Security, logging, rate limiting
│   └── web_ui/       # React SPA
├── packages/         # Domain packages (engine, equity, features, etc.)
├── services/         # Application services (28 services)
├── configs/          # Environment-specific configs
├── tests/            # Test suites
│   ├── unit/
│   ├── integration/
│   ├── property/
│   └── e2e/
├── docs/             # Documentation
└── infra/            # Infrastructure scripts
```

## Key Concepts

1. **Engine** (`packages/engine/`): Core poker game engine supporting 2-6 players
2. **Persistence** (`packages/persistence/`): SQLite-backed hand/session storage
3. **CQRS** (`services/cqrs/`): Command/Query separation for API operations
4. **Features** (`packages/features/`): Feature extraction for ML pipeline
5. **Strategy** (`packages/strategy/`): Mixed strategy with regret matching
