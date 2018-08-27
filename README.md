# Stellar Graph Machine Learning Library

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


## Build Status
|Branch|Build|
|:-----|:----:|
|*master*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=master)|
|*devel*|[![Build status](https://badge.buildkite.com/34d537a018c6bf27cf154aa5bcc287b2e170d6e3391cd40c64.svg)](https://buildkite.com/stellar/stellar-ml?branch=develop)|

## CI

### buildkite integration

Pipeline is defined in `.buildkite/pipeline.yml`

### Docker images

* Tests: Uses the official [python:3.6](https://hub.docker.com/_/python/) image.
* Style: Uses [black](https://hub.docker.com/r/stellargraph/black/) from the `stellargraph` docker hub organisation.