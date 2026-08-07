from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""


def _read_requirements() -> list[str]:
    req_file = ROOT / "requirements.txt"
    if not req_file.exists():
        return []
    return [
        line.strip()
        for line in req_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


setup(
    name="multi-agent-workspace",
    version="3.1.0",
    description="Production Multi-Agent Workspace: RAG + LangGraph Router + SQL AST + Excel/OCR/Table + MCP + PII + RRF + Retry Resilience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AI Platform Engineering",
    packages=find_packages(include=["src", "src.*", "eval", "eval.*"]),
    python_requires=">=3.11",
    install_requires=_read_requirements(),
    extras_require={
        "dev": ["pytest>=8.3.0", "pytest-asyncio>=0.24.0", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "workspace-eval=eval.run_ragas_eval:main",
            "workspace-testset=eval.generate_testset:main",
            "workspace-gates=eval.check_gates:main",
        ],
    },
)
