from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="ml-bot-detection-system",
    version="1.0.0",
    author="grisheet",
    description="ML-powered bot detection system for web traffic classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grisheet/ml-bot-detection-system",
    packages=find_packages(exclude=["tests*", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "joblib>=1.3.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.4.0",
    ],
    extras_require={
        "full": requirements,
        "api": ["fastapi>=0.104.0", "uvicorn[standard]>=0.24.0"],
        "dashboard": ["streamlit>=1.28.0", "plotly>=5.18.0"],
        "xgboost": ["xgboost>=2.0.0"],
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0", "mypy>=1.6.0"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: Log Analysis",
        "Topic :: Security",
    ],
    keywords="bot-detection machine-learning fastapi streamlit cybersecurity",
    entry_points={
        "console_scripts": [
            "botdetect-train=src.train:train_all_models",
            "botdetect-api=api.app:app",
        ]
    },
)
