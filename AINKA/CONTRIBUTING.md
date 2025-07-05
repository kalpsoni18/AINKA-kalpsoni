# Contributing to AINKA

Thank you for your interest in contributing to AINKA! We welcome contributions from everyone in the open source community. This document provides guidelines and standards for contributing to the project.

## 🤝 Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. We are committed to providing a welcoming and inspiring community for all.

## 🚀 Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** thoroughly
6. **Submit** a pull request

## 📋 Development Standards

### Kernel Module (C)
- **Language:** C (C99 standard)
- **License:** GPLv3
- **Style:** Follow Linux kernel coding style
- **Documentation:** Include kernel-doc comments
- **Testing:** Must include kernel module tests

### Daemon/CLI (Rust)
- **Language:** Rust (latest stable)
- **License:** Apache 2.0
- **Style:** Follow Rust formatting standards (`rustfmt`)
- **Documentation:** Include doc comments
- **Testing:** Unit tests required

### General Standards
- **Security:** All code must be security-auditable
- **Performance:** Kernel code must be efficient
- **Error Handling:** Comprehensive error handling required
- **Logging:** Appropriate logging for debugging

## 🔧 Development Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r) rustc cargo git

# Development tools
sudo apt install clang-format rustfmt cargo-audit valgrind
```

### Local Development

```bash
# Clone your fork
git clone https://github.com/yourusername/ainka.git
cd ainka

# Set up pre-commit hooks
cp scripts/pre-commit .git/hooks/
chmod +x .git/hooks/pre-commit

# Build everything
./scripts/build.sh

# Run tests
./scripts/test.sh
```

## 📝 Code Style Guidelines

### C (Kernel Module)
```c
// Use kernel coding style
static int ainka_function(struct ainka_data *data)
{
    int ret;
    
    if (!data)
        return -EINVAL;
    
    ret = some_operation(data);
    if (ret < 0) {
        pr_err("AINKA: Operation failed: %d\n", ret);
        return ret;
    }
    
    return 0;
}
```

### Rust (Daemon/CLI)
```rust
/// AINKA daemon configuration
#[derive(Debug, Clone)]
pub struct AinkaConfig {
    /// Path to kernel interface
    pub kernel_path: String,
    /// Log level
    pub log_level: LogLevel,
}

impl AinkaConfig {
    /// Create a new configuration
    pub fn new() -> Self {
        Self {
            kernel_path: "/proc/ainka".to_string(),
            log_level: LogLevel::Info,
        }
    }
}
```

## 🧪 Testing Guidelines

### Kernel Module Testing
```bash
# Build and load module
cd kernel
make
sudo insmod ainka_lkm.ko

# Test /proc interface
cat /proc/ainka
echo "test" | sudo tee /proc/ainka

# Unload module
sudo rmmod ainka_lkm
```

### Rust Testing
```bash
# Run unit tests
cd daemon
cargo test

# Run integration tests
cargo test --test integration

# Run with coverage
cargo tarpaulin
```

## 📚 Documentation Standards

### Code Documentation
- **Kernel:** Use kernel-doc format
- **Rust:** Use standard Rust doc comments
- **API:** Document all public APIs
- **Examples:** Include usage examples

### Commit Messages
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No security issues introduced
```

## 🔒 Security Guidelines

### Code Review Requirements
- All kernel code must be reviewed by at least one maintainer
- Security-sensitive changes require additional review
- Static analysis must pass
- No known vulnerabilities in dependencies

### Security Best Practices
- Validate all user input
- Use secure coding practices
- Follow principle of least privilege
- Document security considerations

## 🏗️ Architecture Contributions

### Adding New Features
1. **Propose** the feature in an issue
2. **Design** the architecture
3. **Implement** with tests
4. **Document** the changes
5. **Submit** for review

### Plugin Development
- Follow the plugin interface
- Include configuration options
- Provide comprehensive testing
- Document usage and examples

## 🌟 Recognition

Contributors will be recognized in:
- Project README
- Release notes
- Contributor hall of fame
- GitHub contributors page

## 📞 Getting Help

- **Issues:** Use GitHub issues for bugs and feature requests
- **Discussions:** Use GitHub Discussions for questions
- **Documentation:** Check the `docs/` directory
- **Community:** Join our community channels

## 🎯 Contribution Areas

We welcome contributions in these areas:

### High Priority
- Kernel module improvements
- Security enhancements
- Performance optimizations
- Test coverage improvements

### Medium Priority
- New plugins and modules
- Documentation improvements
- CLI enhancements
- Monitoring and metrics

### Low Priority
- UI/UX improvements
- Additional language bindings
- Example applications
- Community tools

## 📄 License

By contributing to AINKA, you agree that your contributions will be licensed under the same license as the component you're contributing to:
- **Kernel code:** GPLv3
- **Userspace code:** Apache 2.0

---

Thank you for contributing to AINKA! Your work helps make Linux systems more intelligent and manageable for everyone. 