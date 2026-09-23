from unittest.mock import patch

import pytest

from stimulusbio.reporting import build_report, capture_environment

# -------------------------
# build_report
# -------------------------

def test_build_report_combines_results_and_provenance():
    results = {"validation": {"status": "PASS"}, "qc": {"library_size": 100}}
    provenance = {"python": "3.11.0"}

    report = build_report(results, provenance)

    assert report == {"results": results, "provenance": provenance}


def test_build_report_performs_no_computation_on_results():
    results = {"anything": object()}
    provenance = {}

    report = build_report(results, provenance)

    assert report["results"]["anything"] is results["anything"]


def test_build_report_does_not_mutate_inputs():
    results = {"validation": {"status": "PASS"}}
    provenance = {"python": "3.11.0"}

    report = build_report(results, provenance)
    report["results"]["extra"] = "should not leak back"
    report["provenance"]["extra"] = "should not leak back"

    assert "extra" not in results
    assert "extra" not in provenance


def test_build_report_rejects_non_dict_results():
    with pytest.raises(TypeError, match="results must be a dict"):
        build_report(["not", "a", "dict"], {})


def test_build_report_rejects_non_dict_provenance():
    with pytest.raises(TypeError, match="provenance must be a dict"):
        build_report({}, provenance="not a dict")


def test_build_report_accepts_empty_dicts():
    report = build_report({}, {})

    assert report == {"results": {}, "provenance": {}}


# -------------------------
# capture_environment
# -------------------------

def test_capture_environment_includes_python_version():
    environment = capture_environment()

    assert "python" in environment
    assert environment["python"].count(".") == 2


def test_capture_environment_includes_all_tracked_packages():
    environment = capture_environment()

    for package_name in (
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ):
        assert package_name in environment
        assert environment[package_name] != ""


def test_capture_environment_reports_missing_package_without_raising():
    from importlib.metadata import PackageNotFoundError

    def fake_version(name):
        if name == "seaborn":
            raise PackageNotFoundError(name)
        return "1.2.3"

    with patch(
        "stimulusbio.reporting.provenance.version",
        side_effect=fake_version,
    ):
        environment = capture_environment()

    assert environment["seaborn"] == "not installed"
    assert environment["numpy"] == "1.2.3"


def test_capture_environment_has_no_dependency_on_analysis_modules():
    import ast
    from pathlib import Path

    source = Path(
        Path(__file__).parent.parent
        / "src"
        / "stimulusbio"
        / "reporting"
        / "provenance.py"
    ).read_text()

    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not any(module.startswith("stimulusbio") for module in imported_modules)
