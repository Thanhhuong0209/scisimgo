# SciSimGo CI/CD Workflows

This directory contains the complete CI/CD pipeline for the SciSimGo project, providing automated testing, building, deployment, and quality assurance.

## Workflow Overview

### 1. **CI/CD Pipeline** (`.github/workflows/ci.yml`)
**Main continuous integration workflow that runs on every push and pull request.**

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Release publications

**Jobs:**
- **Go Testing & Building**: Multi-version testing (Go 1.20, 1.21, 1.22)
- **Python Testing**: Multi-version testing (Python 3.9, 3.10, 3.11)
- **Security Scanning**: Trivy vulnerability scanning
- **Docker Build & Test**: Multi-platform Docker image building
- **Integration Testing**: Docker Compose integration tests
- **Release Creation**: Automated release artifacts
- **Performance Testing**: Benchmark execution
- **Documentation Check**: README and Go documentation validation

**Artifacts:**
- Go binaries for multiple platforms
- Test coverage reports
- Benchmark results
- Docker images

### 2. **Deployment** (`.github/workflows/deploy.yml`)
**Handles deployment to staging and production environments.**

**Triggers:**
- Successful CI pipeline completion
- Manual workflow dispatch

**Jobs:**
- **Staging Deployment**: Automated deployment to staging environment
- **Production Deployment**: Manual deployment to production with version control
- **Rollback**: Automatic rollback on deployment failures

**Features:**
- Environment-specific configurations
- Smoke tests and health checks
- Automated release creation
- Rollback capabilities

### 3. **Dependency Management & Security** (`.github/workflows/dependencies.yml`)
**Automated dependency scanning and security updates.**

**Triggers:**
- Daily scheduled runs (2 AM UTC)
- Manual workflow dispatch

**Jobs:**
- **Go Dependencies**: Vulnerability scanning and license checking
- **Python Dependencies**: Security analysis and dependency updates
- **Docker Security**: Container vulnerability scanning
- **License Compliance**: License validation and reporting
- **Dependency Graph**: Dependency tree generation
- **Security Summary**: Comprehensive security report

**Features:**
- Automated dependency updates via PRs
- Security vulnerability scanning
- License compliance checking
- Dependency graph visualization

### 4. **Quality Assurance** (`.github/workflows/quality.yml`)
**Comprehensive code quality and testing workflow.**

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Daily scheduled runs (6 AM UTC)
- Manual workflow dispatch

**Jobs:**
- **Go Quality**: Linting, formatting, complexity analysis
- **Python Quality**: Multiple linters, type checking, security analysis
- **Performance Testing**: Benchmark execution and analysis
- **Code Metrics**: Complexity and maintainability metrics
- **Quality Summary**: Comprehensive quality report

**Features:**
- Multiple Go and Python versions
- Coverage threshold enforcement (80%)
- Performance benchmarking
- Code complexity analysis

### 5. **Release Management** (`.github/workflows/release.yml`)
**Automated versioning and release creation.**

**Triggers:**
- Git tag pushes (v*)
- Manual workflow dispatch

**Jobs:**
- **Version Management**: Automated version bumping and tagging
- **Build Release**: Multi-platform binary compilation
- **Changelog Generation**: Automated changelog from git commits
- **Release Creation**: GitHub release with artifacts
- **Release Notification**: Status reporting

**Features:**
- Multi-platform builds (Linux, Windows, macOS, ARM64)
- Automated changelog generation
- Checksum verification
- Draft and prerelease support

### 6. **CodeQL Analysis** (`.github/codeql/codeql-analysis.yml`)
**Advanced security and code quality analysis.**

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Weekly scheduled runs (Sunday 2 AM UTC)

**Jobs:**
- **CodeQL Analysis**: Semantic code analysis for Go and Python
- **Security Analysis**: Additional security checks
- **Code Quality Analysis**: Linting and quality checks

**Features:**
- Semantic code analysis
- Security vulnerability detection
- Code quality assessment
- SARIF output for GitHub Security tab

## Configuration Files

### Dependabot (`.github/dependabot.yml`)
- **Automated dependency updates** for Go, Python, GitHub Actions, and Docker
- **Weekly updates** on Mondays at 9 AM UTC
- **Automatic PR creation** with proper labels and assignees
- **Major version update protection** (manual review required)

### CodeQL Configuration (`.github/codeql/codeql-config.yml`)
- **Custom query configuration** for scientific computing security
- **Path exclusions** for test files and generated content
- **Memory and timeout optimization** for large codebases

### Reusable Actions (`.github/actions/setup/action.yml`)
- **Standardized environment setup** for Go and Python
- **Tool installation** for development and testing
- **Caching optimization** for faster workflow execution

## Usage

### Manual Workflow Execution

1. **Go to Actions tab** in your GitHub repository
2. **Select the workflow** you want to run
3. **Click "Run workflow"** button
4. **Configure parameters** as needed
5. **Execute** the workflow

### Common Workflow Parameters

#### Quality Assurance
- `run_all_tests`: Run comprehensive tests including performance
- `check_coverage`: Enforce test coverage thresholds

#### Deployment
- `environment`: Choose staging or production
- `version`: Specify deployment version

#### Release Management
- `version`: Release version (e.g., 1.0.0)
- `release_type`: Major, minor, or patch
- `draft`: Create as draft release
- `prerelease`: Mark as prerelease

#### Dependency Management
- `update_dependencies`: Automatically update dependencies
- `security_only`: Run security scans only

### Environment Variables

The workflows use several environment variables:

```yaml
GO_VERSION: '1.21'
PYTHON_VERSION: '3.11'
COVERAGE_THRESHOLD: '80'
DOCKER_REGISTRY: 'ghcr.io'
```

### Secrets Required

- `GITHUB_TOKEN`: Automatically provided by GitHub
- `ALERT_WEBHOOK_URL`: Optional webhook for notifications

## Monitoring and Alerts

### Workflow Status
- **Green**: All checks passed
- **Yellow**: Some checks failed (non-blocking)
- **Red**: Critical failures (blocking)

### Artifacts
- **Test coverage reports** (30-day retention)
- **Build artifacts** (30-day retention)
- **Benchmark results** (90-day retention)
- **Security reports** (90-day retention)

### Notifications
- **GitHub Actions summary** for each workflow run
- **Release notifications** for successful deployments
- **Security alerts** for vulnerability detection

## Best Practices

### For Developers
1. **Run workflows locally** before pushing
2. **Check test coverage** regularly
3. **Review security reports** monthly
4. **Update dependencies** when security patches are available

### For Maintainers
1. **Monitor workflow performance** and optimize as needed
2. **Review and approve** dependency update PRs
3. **Validate releases** before publishing
4. **Monitor security alerts** and act promptly

### For Contributors
1. **Ensure tests pass** before submitting PRs
2. **Follow code formatting** standards
3. **Add tests** for new functionality
4. **Update documentation** when needed

## Troubleshooting

### Common Issues

1. **Workflow timeout**: Increase timeout values in workflow files
2. **Memory issues**: Optimize resource usage or increase limits
3. **Dependency conflicts**: Review and resolve version conflicts
4. **Build failures**: Check platform-specific build requirements

### Debugging

1. **Enable debug logging** by setting `ACTIONS_STEP_DEBUG=true`
2. **Review workflow logs** for detailed error information
3. **Check artifact uploads** for missing files
4. **Validate environment setup** in reusable actions

## Performance Optimization

### Caching Strategies
- **Go modules**: Cached across workflow runs
- **Python packages**: Cached with pip
- **Docker layers**: Multi-stage build optimization
- **GitHub Actions**: Artifact sharing between jobs

### Parallel Execution
- **Matrix builds** for multiple versions
- **Independent jobs** for faster completion
- **Conditional execution** for optional features
- **Resource optimization** for cost efficiency

## Security Features

### Vulnerability Scanning
- **Go modules**: govulncheck integration
- **Python packages**: safety and bandit scanning
- **Docker images**: Trivy vulnerability scanning
- **Code analysis**: CodeQL semantic analysis

### Access Control
- **Environment protection** for production deployments
- **Required reviewers** for dependency updates
- **Branch protection** for main branches
- **Secret management** for sensitive data

## Future Enhancements

### Planned Features
- **Slack/Teams integration** for notifications
- **Advanced performance profiling** with flame graphs
- **Automated performance regression** detection
- **Machine learning** for code quality prediction
- **Advanced security** scanning with custom rules

### Integration Opportunities
- **External monitoring** systems (Prometheus, Grafana)
- **Issue tracking** automation (Jira, Linear)
- **Documentation** generation (Docusaurus, MkDocs)
- **API testing** with Postman collections

---

For questions or issues with the CI/CD workflows, please:
1. Check the workflow logs for error details
2. Review the troubleshooting section above
3. Create an issue with workflow name and error details
4. Contact the maintainers for complex issues
