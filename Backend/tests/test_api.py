import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure the Backend directory is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cifastapi_mosaic import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "timestamp" in response.json()

def test_unauthorized_conversations():
    # Attempting to fetch conversations without a token should fail
    response = client.get("/conversations")
    assert response.status_code == 401

def test_unauthorized_servers():
    # Attempting to fetch user servers without a token should fail
    response = client.get("/servers")
    assert response.status_code == 401

def test_check_username_invalid():
    # Username check does not require auth, but format should be valid
    response = client.get("/auth/check-username/ab")
    assert response.status_code == 200
    assert response.json()["available"] == False
