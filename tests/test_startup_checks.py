"""Tests for startup dependency checks (spaCy model, Elasticsearch)."""

from unittest.mock import patch, MagicMock

import pytest

from mordecai3.exceptions import (
    SpacyModelError,
    ElasticsearchConnectionError,
    GeonamesIndexError,
)
from mordecai3.geoparse import load_nlp, Geoparser


class TestSpacyModelCheck:
    def test_missing_spacy_model_raises_error(self):
        with patch("mordecai3.geoparse.spacy") as mock_spacy:
            mock_spacy.load.side_effect = OSError("Can't find model")
            with pytest.raises(SpacyModelError):
                load_nlp()

    def test_spacy_error_message_contains_install_command(self):
        with patch("mordecai3.geoparse.spacy") as mock_spacy:
            mock_spacy.load.side_effect = OSError("Can't find model")
            with pytest.raises(SpacyModelError, match="python -m spacy download en_core_web_trf"):
                load_nlp()


class TestElasticsearchChecks:
    """Test ES checks in Geoparser.__init__.

    We mock load_nlp, load_model, and load_hierarchy so the constructor
    only exercises the ES validation path.
    """

    INIT_PATCHES = [
        "mordecai3.geoparse.load_nlp",
        "mordecai3.geoparse.load_model",
        "mordecai3.geoparse.load_hierarchy",
        "mordecai3.geoparse.setup_es_client",
    ]

    def _build(self, es_connected, has_index, check_es=True):
        """Attempt to construct a Geoparser with mocked dependencies."""
        with (
            patch("mordecai3.geoparse.load_nlp") as mock_nlp,
            patch("mordecai3.geoparse.load_model") as mock_model,
            patch("mordecai3.geoparse.load_hierarchy") as mock_hier,
            patch("mordecai3.geoparse.setup_es_client") as mock_setup,
            patch("mordecai3.geoparse.es_is_accepting_connection") as mock_conn,
            patch("mordecai3.geoparse.es_has_geonames_index") as mock_idx,
            patch("mordecai3.geoparse.Search") as mock_search,
        ):
            mock_setup.return_value = MagicMock()
            mock_conn.return_value = es_connected
            mock_idx.return_value = has_index
            mock_nlp.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            mock_hier.return_value = {}

            return Geoparser(check_es=check_es)

    def test_unreachable_es_raises_error(self):
        with pytest.raises(ElasticsearchConnectionError):
            self._build(es_connected=False, has_index=False)

    def test_missing_geonames_index_raises_error(self):
        with pytest.raises(GeonamesIndexError):
            self._build(es_connected=True, has_index=False)

    def test_check_es_false_skips_validation(self):
        # Should not raise even though ES is "down"
        geo = self._build(es_connected=False, has_index=False, check_es=False)
        assert geo is not None
