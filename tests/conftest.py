"""Shared test fixtures for lead scoring project."""

import numpy as np
import pytest
from src.generate_data import generate_accounts, generate_contacts, generate_opportunities


@pytest.fixture
def rng():
    return np.random.default_rng(99)


@pytest.fixture
def sample_accounts(rng):
    return generate_accounts(n=30, rng=rng)


@pytest.fixture
def sample_contacts(sample_accounts):
    return generate_contacts(sample_accounts, rng=np.random.default_rng(100))


@pytest.fixture
def sample_opportunities(sample_accounts, sample_contacts):
    opps, bridge = generate_opportunities(
        sample_accounts, sample_contacts, n_opps=40,
        rng=np.random.default_rng(101)
    )
    return opps, bridge


@pytest.fixture
def sample_bridge(sample_opportunities):
    return sample_opportunities[1]


@pytest.fixture
def sample_opps(sample_opportunities):
    return sample_opportunities[0]
