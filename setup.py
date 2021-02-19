from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="gemini", # Replace with your own username
    version="0.0.1",
    author="Enflame Tech",
    author_email="heng.shi@enflame-tech.com",
    description="model parallel runner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
    ],
    python_requires='>=2.7',
    tests_require=[
        # 'unittest',
    ],
    entry_points={
        "console_scripts": [
            "gemini_python=gemini.bin.gemini_python:main",
        ]
    }
)
