# Pints Matrix

Use the `matrix` tool to run the Pints performance matrix

# Getting Started

When cloning, make sure to add the `--recusive` switch so you clone the Pints
submodule, i.e.

```bash
$ git clone --recursive https://github.com/pints-team/performance-testing
```

After cloning, install the requirements. Note that this will try to install the
Pints submodule, so if you have already installed Pints on your system you
should do this in a separate virtualenv or conda environment.

```bash
$ cd performance-testing
$ pip install -r requirements.txt
```

# Running the tests and plotting results

Please see the help for the `matrix` tool

```bash
$ ./matrix -h
```

