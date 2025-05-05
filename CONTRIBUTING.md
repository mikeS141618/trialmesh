# Contributing to TrialMesh

Thank you for your interest in contributing to **TrialMesh**! We welcome contributions from the community to improve the codebase, documentation, and clinical/technical accuracy of the project.

---

## How to Contribute

### 1. **Reporting Issues**

- **Bugs:** Please open an [issue](https://github.com/mikeS141618/trialmesh/issues) with a clear description, steps to reproduce, and relevant logs or error messages.
- **Feature Requests:** Suggest new features or enhancements by opening an issue and labeling it as a feature request.
- **Clinical/Scientific Feedback:** If you spot a clinical or scientific inaccuracy, please provide references or rationale.

### 2. **Submitting Code**

- **Fork the repository** and create your branch from `main`:
    ```bash
    git checkout -b my-feature
    ```
- **Write clear, modular code** and include tests where appropriate.
- **Follow the existing code style** (see below).
- **Document your changes** in code comments and, if relevant, in the `docs/` directory.
- **Run tests** before submitting:
    ```bash
    pytest tests/
    ```
- **Submit a pull request** with a clear description of your changes and the problem they solve.

### 3. **Code Style**

- Use [PEP8](https://www.python.org/dev/peps/pep-0008/) conventions for Python code.
- Use type hints where possible.
- Write docstrings for all public functions and classes.
- Use descriptive commit messages.

### 4. **Documentation**

- Improve or expand documentation in the `docs/` directory or in code docstrings.
- For major changes, update the relevant markdown files and, if needed, add new notebooks to `notebooks/`.

### 5. **Prompt Engineering**

- If you propose changes to LLM prompts, update `utils/prompt_registry.py` and provide before/after examples in your pull request.
- Include rationale for prompt changes, especially if they affect clinical reasoning or output structure.

### 6. **Testing**

- Add or update tests in the `tests/` directory for new features or bug fixes.
- Ensure all tests pass before submitting a pull request.

### 7. **Clinical/Medical Content**

- If contributing clinical logic, cite relevant guidelines (e.g., NCCN, ESMO) or literature.
- Avoid including any real patient data in pull requests.

---

## Code of Conduct

By participating in this project, you agree to abide by the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).

---

## License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

## Questions?

For questions or discussions, please open an issue or start a discussion thread.

Thank you for helping make TrialMesh better!