
.PHONY: test test-unit test-integration test-performance test-cuopt test-all

# Quick unit tests only
test:
	PYTEST_QUICK=1 pytest tests/unit/ -v

# Unit tests with coverage
test-unit:
	pytest tests/unit/ --cov=src --cov-report=html --cov-report=term

# Integration tests (requires data setup)
test-integration:
	TEST_DATASET=toy pytest tests/integration/ -v

# Performance tests
test-performance:
	pytest tests/performance/ -v -m performance

# cuOpt integration tests (requires cuOpt server)
test-cuopt:
	@if [ -z "$$TEST_CUOPT_URL" ]; then \
		echo "Error: TEST_CUOPT_URL environment variable required"; \
		exit 1; \
	fi
	pytest tests/integration/test_cuopt_integration.py -v -m cuopt

# All tests with real data
test-real:
	@if [ ! -d "$$PRIVATE_DATA_DIR" ]; then \
		echo "Error: PRIVATE_DATA_DIR not found"; \
		exit 1; \
	fi
	TEST_DATASET=real pytest tests/ -v

# Full test suite (CI/CD)
test-all:
	pytest tests/ -v --cov=src --cov-report=xml --junitxml=pytest-results.xml

# Docker test with cuOpt
test-docker:
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

# Clean test artifacts
clean-test:
	rm -rf htmlcov/ .coverage pytest-results.xml .pytest_cache/

# cuOpt integration tests (requires cuOpt server)
test-cuopt:
	@echo "Starting cuOpt integration tests..."
	@if [ -z "$$TEST_CUOPT_URL" ]; then \
		echo "Starting local cuOpt server..."; \
		docker-compose up -d cuopt; \
		sleep 10; \
		export TEST_CUOPT_URL=http://localhost:5000; \
	fi
	TEST_CUOPT_URL=$${TEST_CUOPT_URL:-http://localhost:5000} \
	pytest tests/integration/test_cuopt_integration.py -v -m cuopt

# Full integration test including cuOpt
test-integration-full:
	docker-compose up -d cuopt
	sleep 15
	TEST_CUOPT_URL=http://localhost:5000 pytest tests/integration/ -v