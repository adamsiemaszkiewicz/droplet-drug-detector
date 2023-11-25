# Project Title

Overview of the project.

## Table of Contents
- [Project Title](#project-title)

- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
    - [Azure DevOps](#azure-devops)
    - [Artifacts](#artifacts)
    - [Configs](#configs)
    - [Data](#data)
    - [Docker](#docker)
    - [Environments](#environments)
    - [Notebooks](#notebooks)
    - [Source Code](#source-code)
    - [Testing](#testing)
- [Development](#development)
    - [Source Code Structure](#source-code-structure)
- [Configuration](#configuration)
    - [GitHub Configuration](#github-configuration)
    - [Docker Configuration](#docker-configuration)
    - [Environment Configuration](#environment-configuration)
    - [Formatting, Linting & Type Checking](#formatting-linting--type-checking)
- [License](#license)

## Installation

Before installing the project, ensure that you have the following requirements:

- Python 3.10
- Mamba (for faster and more efficient virtual environments)
- Docker (optional, needed for containerization)
- Git (for version control)

Follow these steps to set up your local environment:

1. **Clone the repository** to your local machine:

    ```bash
    git clone [repository-url]
    cd [repository-name]
    ```

2. **Install Mamba**: If you do not have Mamba installed, you can install it through Conda:

    ```bash
    conda install mamba -n base -c conda-forge
    ```

3. **Create and activate a Conda environment**: Use the provided environment YAML files to create and activate your Conda environment:

    ```bash
    mamba env create -f environments/[environment-name].yaml
    conda activate [environment-name]
    ```

5. **Set up pre-commit hooks** to enforce a variety of standards and validations during each commit:

    ```bash
    pre-commit install
    ```

    To run all pre-commit hooks on all files in the repository, execute:

    ```bash
    pre-commit run --all-files
    ```

6. **Docker setup** (optional): For projects that require Docker, build and run your containers using:

    ```bash
    docker build -t [image-name]:[tag] .
    docker run -it [image-name]:[tag]
    ```

[Back to the top](#project-title)

## Repository Structure

### Azure DevOps

The `.azure-devops` directory contains configurations specific to Azure DevOps features and services to support the project's development workflow.

[Back to the top](#project-title)

### Github

The `.github` directory contains configurations specific to GitHub features and services to support the project's development workflow.

### Artifacts

All experiment related artifacts such as configuration files, model checkpoints, logs, etc. are saved in `artifacts` directory.

[Back to the top](#project-title)


### Configs

The `configs` directory contains configuration YAML files for different machine learning tasks.

[Back to the top](#project-title)


### Data

Store all project related data inside `data` folder.

[Back to the top](#project-title)


### Docker

All Docker-related files necessary for building Docker images and managing Docker containers for the project are located in `docker` directory.

[Back to the top](#project-title)


### Environments

The `environments` directory stores YAML files that define the different Conda environments needed for the project.

[Back to the top](#project-title)


### Notebooks

Jupyter notebooks integral to the project as located in `notebooks` directory..

[Back to the top](#project-title)


### Source Code

The `src` directory contains all source code for the project.

### Testing

The `tests` directory contains various types of automated tests to ensure the codebase works correctly:

- `tests/unit` for unit tests to validate individual pieces of code independently.
- `tests/integration` for integration tests to ensure different code sections work together as intended.
- `tests/e2e` for end-to-end tests that verify complete user workflows.



#### Running Tests

Execute all tests with:

```bash
pytest
```

[Back to the top](#project-title)

## Development

### Azure DevOps Code Structure

A detailed explanation of the layout and purpose of the `.azure-devops` directory contents.

- `.azure-devops/pipelines`: This folder holds the YAML pipeline definitions for building and deploying using Azure DevOps services.
  - `build-aml-environment.yaml` sets up Azure Machine Learning environment needed for running Azure ML tasks
  - `sample-pipeline.yaml` runs a sample Azure Machine Learning task

- `.azure-devops/templates`: Reusable YAML templates with encapsulated functionalities to streamline pipeline creation. The templates include:
  - `install-azure-cli.yaml` for installing the Azure CLI.
  - `configure-aml-extension.yaml` for setting up Azure ML extensions.
  - `connect-to-aml-workspace.yaml` for connecting to an Azure ML workspace within the pipeline.
  - `create-conda-env.yaml` for constructing Conda environments needed for the pipeline's operations.
  - `substitute-env-vars.yaml` for injecting environment variables dynamically into the pipeline process.


### Source Code Structure

A detailed explanation of the layout and purpose of the `src` directory contents.

- `aml`: Azure Machine Learning utilities, components and pipelines.
    - `components`: Contains code for individual Azure Machine Learning components.
        - `sample_component`: An example component, complete with its specification YAML, entrypoints script, options, configuration and custom functions.
    - `pipelines`: Contains code for Azure Machine Learning pipelines.
        - `sample_pipeline`: An example pipeline specification YAML
    - `blob_storage.py`: Azure Blob Storage service allowing to upload and download files and folders.
    - `build_aml_environment.py`: A script to set up the Azure Machine Learning environment.
    - `client.py`: Azure Machine Learning client allowing to interact with AML objects.
    - `environment.py`: Azure Machine Learning environment allowing to create and manage AML environments.

- `common`: Shared utilities and constants used across the project.
    - `consts`: Definitions of constants used throughout the codebase, like Azure-specific constants, directory paths, and extensions.
    - `settings`: Infrasturcture settings storing things such as Azure ML, Azure Blob Storage, cluster & database credentials.
    - `utils`: General utility functions and classes, such as settings management, logging, converters and validators.
- `configs`: Configuration classes for machine learning tasks.
- `machine_learning`: Contains code for machine learning tasks divided into different categories and providing types, configuration and creation..
    - `augmentations`: Data augmentation
    - `callbacks`: Pytorch Lightning training callbacks
    - `classification`: Classification-specific modules
      - `loss_functions`: Loss functions
      - `metrics`: Evaluation metrics
      - `models`: Model architectures
      - `module.py`: Pytorch Lightning module
    - `loggers`: Pytorch Lightning loggers
    - `optimizer`: Optimizer
    - `preprocessing`: Data preprocessing transformations
    - `scheduler`: Learning rate scheduler
    - `trainer`: Pytorch Lightning trainer

[Back to the top](#project-title)



## Configuration

### GitHub Configuration

The `.github` directory contains configurations specific to GitHub features and services to support the project's development workflow.

#### GitHub Actions Workflows

- `workflows`: Includes automation workflows for GitHub Actions. The `ci.yaml` file in this directory configures the continuous integration workflow, which is triggered on push and pull request events to run tests, perform linting, and other checks integral to maintaining code quality and operational integrity.

#### Issue and Pull Request Templates

- `ISSUE_TEMPLATE`: Provides templates for opening new issues on GitHub. The templates ensure that all necessary details are included when contributors report bugs (`bug_report.md`) or propose new features (`feature_request.md`). Use these templates to create issues that are consistent and informative.

[Back to the top](#project-title)

### Docker Configuration

The `docker` directory is intended to house all Docker-related files necessary for building Docker images and managing Docker containers for the project. This includes:

- **Dockerfiles**: Each Dockerfile contains a set of instructions to assemble a Docker image. Dockerfiles should be named with the convention `Dockerfile` or `Dockerfile.<environment>` to denote different setups, such as development, testing, or production environments.

- **docker-compose files**: For projects that run multiple containers that need to work together, `docker-compose.yml` files define and run multi-container Docker applications. With Compose, you use a YAML file to configure your application's services and create and start all the services from your configuration with a single command.

- **Configuration Scripts**: Any scripts that aid in setting up, building, or deploying Docker containers, such as initialization scripts or entrypoint scripts, belong here.

- **Environment Files**: `.env` files that contain environment variables necessary for Docker to run or for Dockerized applications to operate correctly can be placed in this directory. These files should not contain sensitive information and should be excluded from version control if they do.

As the project develops, ensure that you populate the `docker` directory with these files and provide documentation on their purpose and how they should be used. This could include instructions on how to build images, start containers, and manage containerized environments effectively.

[Back to the top](#project-title)

### Environment Configuration

The `environments` directory contains configuration files that define the different environments needed for the project. These files are essential for ensuring that the project runs with the correct versions of its dependencies and in a way that's consistent across different setups.

- **Conda Environment Files**: YAML files that specify the packages required for a conda environment. Environment YAML files should have the same name as the project they relate to.

- **Infrastructure Configuration**: The `infra.yaml` file might include configurations for setting up the infrastructure as a code, which can be particularly useful when working with cloud services or when you want to automate the setup of your project's infrastructure.

[Back to the top](#project-title)

### Formatting, Linting & Type Checking

The pre-commit hooks will now automatically check each file when you attempt to commit them to your git repository. If any hooks make changes or fail, fix the issues, and try committing again.

Here are the hooks configured for this project:

- `flake8`: Lints Python source files for coding standard violations, complexity, and style issues.
- `black`: Formats Python code to ensure consistent styling.
- `isort`: Sorts Python imports alphabetically within respective sections and by type.
- `mypy`: Checks type hints and enforces type checking on your code.
- `pytest`: Runs automated tests to make sure new changes do not break the functionality.

[Back to the top](#project-title)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

[Back to the top](#project-title)
