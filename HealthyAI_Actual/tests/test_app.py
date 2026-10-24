import app


def test_screening_band_review():

    assert (
        app.screening_band(
            "mel",
            0.91,
        )
        == "REVIEW"
    )


def test_screening_band_lower():

    assert (
        app.screening_band(
            "nv",
            0.91,
        )
        == "LOWER-CONCERN"
    )


def test_screening_band_uncertain():

    assert (
        app.screening_band(
            "nv",
            0.41,
        )
        == "UNCERTAIN"
    )


def test_model_status():

    status = (
        app.get_model_status()
    )

    assert "ready" in status
    assert "model" in status
    assert "classes" in status
    assert "inference" in status


def test_health_endpoint():

    client = (
        app.app.test_client()
    )

    response = client.get(
        "/api/health"
    )

    assert response.status_code == 200

    assert (
        response.json["ok"]
        is True
    )


def test_stats_endpoint():

    client = (
        app.app.test_client()
    )

    response = client.get(
        "/api/stats"
    )

    assert response.status_code == 200

    assert "total" in response.json
    assert "bands" in response.json